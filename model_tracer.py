
import torch
import torch.nn as nn
import torch.nn.functional as F


orig_module_forward = torch.nn.Module.__call__
format_id = lambda x : ("00000000"+str(x))[-8:]
module_names = [m for m in dir(nn) if isinstance(getattr(nn, m), type) and issubclass(getattr(nn, m), nn.Module) and m != "Module"]
keep_modules = ['Sequential', 'ModuleList', "Embedding"]
# to_exclude_modules = ["Identity", "Conv2d"] 
# to_exclude_modules = ["Identity"] 
to_exclude_modules = [m for m in module_names if m not in keep_modules]
# TODO the danger here is when you exclude a module, no opportunity for creating RecorderTensor, happens in SD embedding,
# maybe others
# Identity causes problem bc no ops performed, just returns, breaks apparatus. 
# NOTE TODO empty modules eg Identity break apparatus bc __torch_function__ never called 
# NOTE excluding modules speeds things up substantially
# NOTE including all except Identity doesn't fix coat_mini extra nodes
# swinv2 droppath, empty module

# something about the parenthood switch made SD layout not as good. still have safe copy of model_tracer
# this is the one that uses global gc to track context parenthood. It works better for CLIP, and i like it better,
# but it's creating a small layout something w SD. Fix this
# HERE

class Container:
    def __init__(self):
        self.reset()

    def reset(self):
        self.edge_id_ix = 0; self.node_id_ix = 0
        self.nodes = []; self.modules = []; self.edges = {}
        self.context = []
        self.execution_counter = 0

    def create_new_edge(self, t, is_global_input=False):
        # Note this also modifies the tensor passed in
        edge_id = format_id(self.edge_id_ix)
        edge = {"shape":t.shape,
                "edge_id":edge_id,
                "is_global_input":is_global_input}
        self.edges[edge_id] = edge
        t.edge_id = edge_id
        self.edge_id_ix += 1
        return edge

    def create_new_node(self, name, node_type):
        # have to append manually
        node = {"name": name,
                "node_type":node_type,
                "inbound_edges":[],
                "outbound_edges":[],
                "input_shapes":[],
                "output_shapes":[],
                "node_id":format_id(self.node_id_ix)}
        self.node_id_ix += 1
        return node

gc = Container()

def traverse_arbitrary_thing(thing, fn_to_apply, target_types):
    # swaps out things, doesn't mod in place bc sometimes need to eg swap in RecorderTensor.
    # though of course objects may be modded in place then returned. fn_to_apply has to return object
    if type(thing) in target_types:
        return fn_to_apply(thing)
    elif type(thing) in [list, tuple]:
        prev_type = type(thing); thing = list(thing)
        for i,subthing in enumerate(thing): thing[i] = traverse_arbitrary_thing(subthing, fn_to_apply, target_types)
        thing = prev_type(thing)
        return thing
    elif type(thing)==dict: # so can use w kwargs in addition to args
        for k,v in thing.items():
            thing[k] = traverse_arbitrary_thing(v, fn_to_apply, target_types)
        return thing
    elif hasattr(thing, "__dict__"): # custom objects. CLIP needs this bc of a huggingface output object in Transformers library
        # print("tranversing a custom object") why is this happening so often?
        attributes = list(thing.__dict__.keys())
        for a in attributes:
            v = getattr(thing, a)
            v = traverse_arbitrary_thing(v, fn_to_apply, target_types)
            setattr(thing, a, v)
        return thing
    else:
        return thing

import copy


# This works, but complicates. For now, still trying to work on our existing layout problems. 
# i do want this back in, but later
orig_name_list = [
    # "as_tensor", "from_numpy", "zeros", "zeros_like",
    # "ones", "ones_like", "arange", "range", "linspace",
    # "logspace", "eye", "empty", "empty_like", "full",
    # "full_like", "complex", "heaviside", "bernoulli",
    # "multinomial", "normal", "poisson", "rand", "rand_like",
    # "randint", "randint_like", "randn", "randn_like",
    # "randperm", # from torchview
    # "FloatTensor", # RG
    # "HalfTensor", 
    # "IntTensor", 
    # "LongTensor", 
    # "BoolTensor"
    # # "Tensor", # this breaks things
]
_orig_op_list = [getattr(torch, name) for name in orig_name_list]



not_respath_modules = ["StyleVectorizer"]

class Recorder:

    def __enter__(self) -> None:
        gc.reset() # always fresh container w each wrapped block

        ##############################################################
        # New module-forward, wrapping normal forward w apparatus to capture the graph

        def new_module_forward(mod, *args, **kwargs):
            # print("\nFORWARD MOD", mod.__class__.__name__, [type(a) for a in args], kwargs)

            if mod.__class__.__name__ in to_exclude_modules:
                out = orig_module_forward(mod, *args, **kwargs)
                return out

            # print("\n\nstart ",  mod.__class__.__name__)
            # Module node
            module_node = gc.create_new_node(name=mod.__class__.__name__, node_type="module")
            parent_ops_wout_module = copy.deepcopy(gc.context)
            module_node["parent_ops"] = parent_ops_wout_module
            module_name_id = mod.__class__.__name__+"-"+module_node["node_id"]
            gc.context.append(module_name_id)
            parent_ops_including_this_module = copy.deepcopy(gc.context)

            # Input edges, grab or create
            # everything after here should be RecorderTensor
            child_input_nodes = []
            added_ids = []
            def enrich_tensor_arg(a):
                nonlocal child_input_nodes
                is_global_input = False
                if type(a)==RecorderTensor: # tensor is coming from a prev op
                    pass
                elif type(a)==torch.Tensor: # create new edge. This is an input tensor
                    a = a.as_subclass(RecorderTensor) # Cast torch.Tensor as RecorderTensor
                    _ = gc.create_new_edge(a, is_global_input=True) # marks id on `a``
                    is_global_input = True

                if True: 
                    #id(a) not in added_ids: This was causing problems actually. 
                    # Can look into further if you want, but made DLA and vovnet break. 
                    added_ids.append(id(a))
                    # input node
                    child_input_node = gc.create_new_node(name="in", node_type="input")
                    child_input_node["is_input"] = True
                    if not is_global_input: child_input_node["inbound_edges"].append(a.edge_id) 
                    # global inputs, consider tensor to originate here, though it actually came in on args
                    # otherwise these are getting inbound edge marked, but there is no inbound edge. Also and especially
                    # this results in modules being marked as input bc of this 'edge' coming in, module input assembled from
                    # these input nodes below
                    if is_global_input: child_input_node["is_global_input"] = True
                    new_a = a.clone() # can't change the underlying tensor, it may be used elsewhere eg on another branch
                    _ = gc.create_new_edge(new_a) # overrides edge_id of tensor of `a` and creates duplicate edge after the input node
                    child_input_node["outbound_edges"].append(new_a.edge_id)
                    child_input_node["parent_ops"] = parent_ops_including_this_module
                    child_input_node["input_shapes"].append(str(list(a.shape)))
                    child_input_node["output_shapes"].append(str(list(a.shape)))

                    child_input_nodes.append(child_input_node)
                    if hasattr(a, "input_nodes"): new_a.input_nodes = a.input_nodes
                    
                    # passing nodes themselves in, they'll be marked in the next op. If not marked by op,
                    # they won't be appended and will disappear, don't want stranded input ops when something
                    # is passed in but not used. Need list here bc when nested, will get two or more input nodes in a row
                    # before we get an op to mark them as used
                    child_input_node["has_been_used"] = False
                    if not hasattr(new_a, "input_nodes"): new_a.input_nodes = []
                    new_a.input_nodes.append(child_input_node)
                else:
                    new_a = a
                    print('got duplicate tensors in input bundle')

                return new_a

            args = traverse_arbitrary_thing(args, enrich_tensor_arg, [RecorderTensor, torch.Tensor])
            kwargs = traverse_arbitrary_thing(kwargs, enrich_tensor_arg, [RecorderTensor, torch.Tensor])

            # Run normal forward, using our enriched args. Functions and submodules called in here.
            out = orig_module_forward(mod, *args, **kwargs)

            # Pad w output nodes off each output
            added_nodes_ids = [] # SD was getting duplicates of same tensor inside output bundle
            module_outbound_edges = []
            def add_output_node(t):
                if (id(t) not in added_nodes_ids): 
                    # NOTE pay attn to this, may want to remove the id() check. SD benefits, it removes extra output nodes, 
                    # but doing the same on inputs was breaking DLA and vovnet
                    added_nodes_ids.append(id(t))

                    ################################
                    # Output node on inside of module
                    child_output_node = gc.create_new_node(name="out", node_type="output")
                    child_output_node["is_output"] = True
                    child_output_node["execution_counter"] = gc.execution_counter
                    child_output_node["inbound_edges"].append(t.edge_id)
                    _ = gc.create_new_edge(t)
                    # overrides edge_id of tensor of `a` and creates duplicate edge after the input node.
                    # Overrides existing tensor, which is ok in this case bc nothing else will use it
                    child_output_node["outbound_edges"].append(t.edge_id)
                    # module_node["outbound_edges"].append(t.edge_id)
                    child_output_node["parent_ops"] = parent_ops_including_this_module

                    child_output_node["input_shapes"].append(str(list(t.shape)))
                    child_output_node["output_shapes"].append(str(list(t.shape)))

                    module_outbound_edges.append(t.edge_id)

                    if mod.__class__.__name__ in not_respath_modules:
                        child_output_node["not_respath"] = True

                    gc.nodes.append(child_output_node)


                    ###################################
                    # Separate tensor node on outside of module
                    tensor_output_node = gc.create_new_node(name="mod_out", node_type="mod_out")
                    tensor_output_node["inbound_edges"].append(t.edge_id)
                    _ = gc.create_new_edge(t)
                    # overrides edge_id of tensor of `a` and creates duplicate edge after the input node.
                    # Overrides existing tensor, which is ok in this case bc nothing else will use it
                    tensor_output_node["outbound_edges"].append(t.edge_id)
                    tensor_output_node["parent_ops"] = parent_ops_wout_module # same parentage level as module, ie outside module

                    tensor_output_node["input_shapes"].append(str(list(t.shape)))
                    tensor_output_node["output_shapes"].append(str(list(t.shape)))
                    tensor_output_node["execution_counter"] = gc.execution_counter

                    gc.nodes.append(tensor_output_node)
                    ###################################



                    # marking input nodes to indicate they should be kept. Passing along counts as using.
                    if hasattr(t, "input_nodes"): 
                        for n in t.input_nodes: n["has_been_used"] = True

                else:
                    print('got duplicate tensors in output bundle')
                return t
            out = traverse_arbitrary_thing(out, add_output_node, [RecorderTensor])

            # only save input nodes when they've been used
            # if haven't been used, don't include in nodes, and don't add inbound edge for module either
            # this is somewhat not true to reality, bc these tensors are really being passed into the module,
            # so it's true that the module will be executed subsequent to those tensors, but it was messy.
            # can consider keeping them in, as it's potentially helpful for debugging a nn, but will need some work
            # Outputting a node counts as using, eg coat_mini
            for c in child_input_nodes:
                if c["has_been_used"]: 
                    # if True: 
                    gc.nodes.append(c)
                    t = c["inbound_edges"]
                    assert len(t) <= 1, "each input node represents a single tensor" # when global input, may be empty
                    module_node["inbound_edges"] += t # comment out when using input nodes (below)

                    # ###################################
                    # # Separate tensor node on outside of module
                    # if len(t)>0: # if global input, don't make
                    #     tensor_input_node = gc.create_new_node(name="mod_in", node_type="mod_in")
                    #     tensor_input_node["inbound_edges"] = copy.deepcopy(c["inbound_edges"])
                    #     new_edge = tensor_input_node["node_id"] + "-" + c["node_id"]
                    #     tensor_input_node["outbound_edges"] = [new_edge]
                    #     c["inbound_edges"] = [new_edge] # override prev
                    #     tensor_input_node["parent_ops"] = parent_ops_wout_module # same parentage level as module, ie outside module

                    #     tensor_input_node["input_shapes"] = c["input_shapes"]
                    #     tensor_input_node["output_shapes"] = c["output_shapes"]

                    #     gc.nodes.append(tensor_input_node)

                    #     module_node["inbound_edges"] += [new_edge]
                    # ###################################

            # doing this after. Patches a bug where we add multiple output nodes bc same tensor shows up twice in
            # output bundle. From SD. Would also fix it to not add multiple times, i think. Doing this after adds
            # them in sequence, so we get two outs in a row but at least they connect correctly. If do above, then all
            # a tangle and outputs don't work. What we should really do is use id() to not add outputs or inputs more 
            # than once for each input
            # def add_module_outbound_edges(t):
            #     module_node["outbound_edges"].append(t.edge_id)
            #     return t
            # out = traverse_arbitrary_thing(out, add_module_outbound_edges, [RecorderTensor])
            module_node["outbound_edges"] += module_outbound_edges

            gc.modules.append(module_node)

            gc.context.pop()
            # print("end ",  mod.__class__.__name__)


            return out

        setattr(torch.nn.Module, "__call__", new_module_forward)

        #######################################
        # Creation ops
        def creation_ops_wrapper(_orig_op):
            def _func(*args, **kwargs):
                print("making freshie!", gc.context)
                input_tensor = _orig_op(*args, **kwargs)
                input_tensor = input_tensor.as_subclass(RecorderTensor) # Cast torch.Tensor as RecorderTensor
                _ = gc.create_new_edge(input_tensor, is_global_input=True) # marks id on `a``
                                             
                input_node = gc.create_new_node(name="fresh in", node_type="input")
                input_node["is_input"] = True
                input_node["is_freshly_made"] = True
                input_node["is_global_input"] = True
                input_node["outbound_edges"].append(input_tensor.edge_id)

                input_node["parent_ops"] = copy.deepcopy(gc.context)
                input_node["input_shapes"].append(str(list(input_tensor.shape)))
                input_node["output_shapes"].append(str(list(input_tensor.shape)))
                gc.nodes.append(input_node)
            
                return input_tensor
            return _func
        
        for name, op in zip(orig_name_list, _orig_op_list):
            setattr(
                torch, name, creation_ops_wrapper(op)
            )
        # end new module-forward
        ###############################################

    def __exit__(self, exc_type, exc_value, exc_traceback):

        setattr(torch.nn.Module, "__call__", orig_module_forward)
        
        for name, op in zip(orig_name_list, _orig_op_list): setattr(torch, name, op)

not_respath_fns = ["argmax"]

class RecorderTensor(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # # NOTE beware this. Things not passed along if things are skipped
        # if func.__name__ in ["clone", "__get__", "dim", "__getitem__"]:# + to_remove_names: # If add these in, edge_ids not passing through
        # print(func.__name__)

        # # shouldn't have any torch.Tensors here. Getting lots in cat, bn, others
        # input_has_torch_tensors = False
        # def check_is_torch_tensor(t):
        #     nonlocal input_has_torch_tensors
        #     input_has_torch_tensors = True
        #     return t
        # args = traverse_arbitrary_thing(args, check_is_torch_tensor, [torch.Tensor])
        # if input_has_torch_tensors: print("input to fn has torch.Tensor rather than RecorderTensor", func.__name__)

        # Don't do recording during some ops. But make sure metadata stays attached.
        if func.__name__ in ["clone", "__get__", "dim", "size", "__repr__"]:# __get_item__ here causes edge to not pass through
            input_tensors = []
            def gather_tensors(t): 
                nonlocal input_tensors
                input_tensors.append(t)
                return t
            args = traverse_arbitrary_thing(args, gather_tensors, [RecorderTensor])
            edge_id = None
            assert not len(input_tensors) > 1
            if len(input_tensors)==1:
                if hasattr(input_tensors[0], "edge_id"): #TODO this is a bit sloppy
                    edge_id = input_tensors[0].edge_id
                    
            # size called in tnt; __repr__ from SD
            out = super().__torch_function__(func, types, args, kwargs)

            if edge_id is not None:
                def attach_metadata(t):
                    if not hasattr(t, "edge_id"):
                        t.edge_id = edge_id
                    return t
                out = traverse_arbitrary_thing(out, attach_metadata, [RecorderTensor])
            
            # def check_has_metadata(t):
            #     assert hasattr(t, "edge_id"), f"wtf no edge id on output?"
            #     return t
            # out = traverse_arbitrary_thing(out, check_has_metadata, [RecorderTensor])

            return out

        # Add function node
        function_node = gc.create_new_node(func.__name__, "function")
        function_node["parent_ops"] = copy.deepcopy(gc.context)
        if func.__name__ in not_respath_fns: function_node["not_respath"] = True
        gc.execution_counter += 1

        # some fns not getting output node (bool, setitem, numpy) bc they have no tensors. But they may be output nodes,
        # so need this marked
        function_node["execution_counter"] = gc.execution_counter
        
        # mark inputs and parent_ops of fn node
        def process_input_tensor(t):
            assert hasattr(t, "edge_id"), f"wtf no edge id on input? {t.shape}"

            function_node["inbound_edges"].append(t.edge_id)
            function_node["input_shapes"].append(str(list(t.shape)))

            if hasattr(t, "input_nodes"):  # marking input nodes to indicate they should be kept
                for n in t.input_nodes: n["has_been_used"] = True
            return t
        args = traverse_arbitrary_thing(args, process_input_tensor, [RecorderTensor])
        kwargs = traverse_arbitrary_thing(kwargs, process_input_tensor, [RecorderTensor])

        # normal fn. Note we're not modifying the input args, so this could be earlier
        out = super().__torch_function__(func, types, args, kwargs)

        ######################################
        # Add new edge for output tensors, cache info on tensors themselves
        def process_output(t):
            out_edge = gc.create_new_edge(t)
            function_node["outbound_edges"].append(out_edge["edge_id"])
            function_node["output_shapes"].append(str(list(t.shape)))
            # if t.shape == torch.Size([1, 1000]): print("output tensor has shape [1,1000]") debugging volos

            ###################################
            # Separate tensor nodes
            tensor_output_node = gc.create_new_node(name="fn_out", node_type="fn_out")
            tensor_output_node["inbound_edges"].append(t.edge_id)
            _ = gc.create_new_edge(t)
            # overrides edge_id of tensor of `a` and creates duplicate edge after the input node.
            # Overrides existing tensor, which is ok in this case bc nothing else will use it
            tensor_output_node["outbound_edges"].append(t.edge_id)
            tensor_output_node["parent_ops"] = copy.deepcopy(gc.context) # same parentage level as fn

            tensor_output_node["input_shapes"].append(str(list(t.shape)))
            tensor_output_node["output_shapes"].append(str(list(t.shape)))
            tensor_output_node["execution_counter"] = gc.execution_counter
            
            gc.nodes.append(tensor_output_node)
            ###################################

            return t
        traverse_arbitrary_thing(out, process_output, [RecorderTensor])

        gc.nodes.append(function_node)

        return out

# Toy models for debugging
    
class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, 3, padding=1)

    def forward(self, x):
        _x = F.relu(self.conv1(x))
        x = x + _x
        return x

class ParallelBlocks(nn.Module):
    def __init__(self):
        super(ParallelBlocks, self).__init__()
        self.catconv = nn.Conv2d(6, 3, 3, padding=1)

    def forward(self, x_tuple):
        x1, x2 = x_tuple
        catted = torch.cat([x1, x2], dim=1)
        catted = self.catconv(catted)
        x1 = x1 + catted
        x2 = torch.cat([x2, catted], dim=1)
        return (x1, x2)

class ElDoble(nn.Module):
    def __init__(self):
        super(ElDoble, self).__init__()
        self.parallel = ParallelBlocks()

    def forward(self, x):
        x_tuple = (x, F.relu(x))
        x1, x2 = self.parallel(x_tuple)
        x = torch.cat([x1, x2], dim=1)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.stem = nn.Conv2d(3, 32, 3)
        self.block1 = ResBlock()
        self.block2 = ResBlock()

    def forward(self, x):
        x = F.relu(self.stem(x))
        x = self.block1(x)
        x = self.block2(x)
        return x
