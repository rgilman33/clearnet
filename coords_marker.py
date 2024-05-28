        
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

intx = lambda l1, l2 : len([i for i in l1 if i in l2]) > 0 

def get_input_children(children):
    all_outbound_edges = []
    for c in children: all_outbound_edges += c["outbound_edges"]
    input_children = [c for c in children if (len([e for e in c["inbound_edges"] if e not in all_outbound_edges])>0) or 
                                            (len(c["inbound_edges"])==0)]
    return input_children

def get_output_children(children):
    # output_children = [c for c in children if intx(c["outbound_edges"], g["outbound_edges"])] this was missing the final output node?
    all_inbound_edges = []
    for c in children: all_inbound_edges += c["inbound_edges"]
    output_children = [c for c in children if (len([e for e in c["outbound_edges"] if e not in all_inbound_edges])>0) or 
                                                (len(c["outbound_edges"])==0)]
    return output_children

def get_downstream_ops(op, peer_ops): 
    return [p for p in peer_ops if intx(op["outbound_edges"], p["inbound_edges"])]

def get_upstream_ops(op, peer_ops): 
    return [p for p in peer_ops if intx(p["outbound_edges"], op["inbound_edges"])]


def get_all_up_or_downstream_nodes(op, ops, un_or_dn_getter_fn):
    # Initialize all nodes as not traced
    for n in ops: n['traced'] = False

    def _trace(n):
        # Mark the current node as traced
        n['traced'] = True
        # Find downstream nodes
        next_nodes = un_or_dn_getter_fn(n, ops)
        for nn in next_nodes:
            if not nn['traced']: _trace(nn)

    # Start tracing from the given node
    _trace(op)
    # Collect all traced nodes
    traced_nodes = [n for n in ops if n['traced']]
    # cleanup
    for o in ops: 
        if o.get("traced") is not None: del o["traced"]
    return traced_nodes

import time
class Timer():
    def __init__(self, timer_name):
        self.init_time = time.time()
        self.results = {}
        self.last_milestone_time = self.init_time
        self.timer_name = timer_name

    def log(self, milestone):
        t = time.time() 
        self.results[f"timing/{milestone}"] = t - self.last_milestone_time
        self.last_milestone_time = t
    
    def finish(self):
        self.results[f"timing/{self.timer_name}"] = time.time() - self.init_time
        return self.results

def pretty_print(d):
    for k,v in d.items():
        print(f"{k.split('/')[-1]}: {round(v, 3)}")

import json

class ModelCoordsMarker():
    def __init__(self, nodes, modules, graph_name):
        self.modules = modules 

        # PositionalEncodingFourier in xcit_medium_24_p8_224 and edgenext_base doesn't record any input or output
        # tensors. Input is a shape tuple, presumably instructions to create the output, which is a generator of some type.
        # I'd like to be capturing this somehow. Note it also doesn't call any torch fns in the module. 
        empty_modules = [m for m in self.modules if len(m["inbound_edges"])+len(m["outbound_edges"])==0] 
        if len(empty_modules)>0:
            print(f"removing {len(empty_modules)} empty modules: {[m['name'] for m in empty_modules]}")
            self.modules = [m for m in self.modules if m["node_id"] not in [mm["node_id"] for mm in empty_modules]]
        self.nodes = nodes

        print(f"there are {len(self.nodes)} nodes and {len(self.modules)} modules")

        self.graph_name = graph_name
        self.timer = Timer("marking coords")

    def compile(self):
        self.add_root_module()
        self.make_nodes_lookup()
        self.mark_dist_from_end()
        self.timer.log("mark dist from end")
        self.mark_dist_from_start()
        self.timer.log("mark dist from start")
        self.denote_module_children()
        self.timer.log("denote module children")
        self.nestify()
        self.save()
        pretty_print(self.timer.finish())

    def save(self):
        filename = f"model_specs/{self.graph_name}.json"

        with open(filename, 'w') as file:
            json.dump(self.nested_hierarchy, file, indent=4)
        
        # update index
        model_specs_overview_filename = "model_specs/model_specs_overview.json"

        with open(model_specs_overview_filename, 'r') as file:
            model_specs_overview = json.load(file)
            
        model_specs_overview[self.graph_name] = filename

        with open(model_specs_overview_filename, 'w') as file:
            json.dump(model_specs_overview, file, indent=4)

            
    def add_root_module(self):
        for n in self.nodes: n["parent_ops"] = ["Root-ROOT"] + n["parent_ops"]
        for mod in self.modules: mod["parent_ops"] = ["Root-ROOT"] + mod["parent_ops"]
        self.modules.append({
            'name': 'Root',
            'node_type': 'module',
            'inbound_edges': [],
            'outbound_edges': [],
            'node_id': 'ROOT',
            'parent_ops': []
        })

    def make_nodes_lookup(self):
        self.nodes_lookup = {}
        for n in self.nodes:
            self.nodes_lookup[n["node_id"]] = n

        for n in self.nodes:
            n["uns"] = []
            n["dns"] = []

        all_edges = []
        for n in self.nodes:
            for e in n["inbound_edges"] + n["outbound_edges"]:
                if e not in all_edges:
                    all_edges.append(e)
        
        for e in all_edges:
            un = None; dns = []
            for n in self.nodes:
                if e in n["outbound_edges"]:
                    un = n
                elif e in n["inbound_edges"]: # NOTE this doesn't allow recursion
                    dns.append(n)

            if un is not None:
                for dn in dns:
                    un["dns"].append(dn["node_id"])
                    dn["uns"].append(un["node_id"])

        self.timer.log("making nodes lookup")
    

    def mark_dist_from_end(self): # global

        DIST_FROM_END_PLACEHOLDER = 1e6
        for n in self.nodes: 
            n["dist_from_end_global"] = DIST_FROM_END_PLACEHOLDER
            n["dist_from_end_originator"] = None

        def mark_and_return_prev_unmarked_nodes(nodes_arr, dist_from_end_originator):
            prev_nodes = []
            for n in nodes_arr:
                # upstream_nodes = get_upstream_ops(n, self.nodes)
                upstream_nodes = [self.nodes_lookup[nid] for nid in n["uns"]]
                v = n["dist_from_end_global"] + 1 
                for un in upstream_nodes: 
                    if (un.get("dist_from_end_originator")==None) or \
                        ((un.get("dist_from_end_originator")!=dist_from_end_originator) and (un["dist_from_end_global"] < v)): 
                        # If node is unseen, mark it. If it's been seen, then mark it if the current value
                        # is lower than the new value. This will happen if eg mark from a shorter endpoint first, then
                        # have to override it. The result here is "distance to furthest endpoint"
                        un["dist_from_end_global"] = v
                        un["dist_from_end_originator"] = dist_from_end_originator
                        prev_nodes.append(un)

            return prev_nodes
        
        output_nodes_global = get_output_children(self.nodes) 
        self.timer.log("get output children")

        for o in output_nodes_global:
            if o.get("execution_counter") is None: # Should this actually just be x?
                o["execution_counter"] = 0
                print("no execution counter for global output node ", o["name"])
        output_nodes_global.sort(key=lambda x: x["execution_counter"], reverse=True) 

        # output w most downstream ops is primary
        for o in output_nodes_global: o["is_output_global"] = True
        
        print("n global output nodes", len(output_nodes_global))
        for i,n in enumerate(output_nodes_global): 
            n["dist_from_end_originator"] = i
            n["dist_from_end_global"] = 0
            prev_nodes = mark_and_return_prev_unmarked_nodes([n], i)
            while len(prev_nodes) > 0:
                prev_nodes = mark_and_return_prev_unmarked_nodes(prev_nodes, i)
        n_nodes_unmarked = len([n for n in self.nodes if n["dist_from_end_global"]==DIST_FROM_END_PLACEHOLDER])
        print(f"There are {n_nodes_unmarked} nodes left unmarked for 'dist to end'")

    # depth first not faster? a tiny bit slower in fact. surprised by this.
    # def mark_dist_from_end(self): # global

    #     DIST_FROM_END_PLACEHOLDER = 1e6
    #     for n in self.nodes: 
    #         n["dist_from_end_global"] = DIST_FROM_END_PLACEHOLDER
    #         n["dist_from_end_originator"] = None

    #     def mark_prev_dist_to_end(base_node, dist_from_end_originator):
    #         upstream_nodes = get_upstream_ops(base_node, self.nodes)
    #         v = base_node["dist_from_end_global"] + 1 
    #         for un in upstream_nodes: 
    #             if (un.get("dist_from_end_originator")==None) or \
    #                 ((un.get("dist_from_end_originator")!=dist_from_end_originator) and (un["dist_from_end_global"] < v)): 
    #                 # If node is unseen, mark it. If it's been seen, then mark it if the current value
    #                 # is lower than the new value. This will happen if eg mark from a shorter endpoint first, then
    #                 # have to override it. The result here is "distance to furthest endpoint"
    #                 un["dist_from_end_global"] = v
    #                 un["dist_from_end_originator"] = dist_from_end_originator
                    
    #                 mark_prev_dist_to_end(un, dist_from_end_originator)
        
    #     output_nodes_global = get_output_children(self.nodes) 
    #     self.timer.log("get output children")

    #     for o in output_nodes_global:
    #         if o.get("execution_counter") is None: # Should this actually just be x?
    #             o["execution_counter"] = 0
    #             print("no execution counter for global output node ", o["name"])
    #     output_nodes_global.sort(key=lambda x: x["execution_counter"], reverse=True) 

    #     # output w most downstream ops is primary
    #     for o in output_nodes_global: o["is_output_global"] = True
        
    #     print("n global output nodes", len(output_nodes_global))
    #     for i,n in enumerate(output_nodes_global): 
    #         n["dist_from_end_originator"] = i
    #         n["dist_from_end_global"] = 0
    #         mark_prev_dist_to_end(n, i)

    #     n_nodes_unmarked = len([n for n in self.nodes if n["dist_from_end_global"]==DIST_FROM_END_PLACEHOLDER])
    #     print(f"There are {n_nodes_unmarked} nodes left unmarked for 'dist to end'")


    def mark_dist_from_start(self): # global
        DIST_FROM_START_PLACEHOLDER = 1e6
        for n in self.nodes: n["dist_from_start_global"] = DIST_FROM_START_PLACEHOLDER
        def mark_and_return_next_unmarked_nodes(nodes_arr, dist_from_start_originator):
            next_nodes = []
            for n in nodes_arr:
                # downstream_nodes = get_downstream_ops(n, self.nodes) # this is what was making so time-consuming
                downstream_nodes = [self.nodes_lookup[nid] for nid in n["dns"]]

                for dn in downstream_nodes: 
                    if dn.get("dist_from_start_global")==DIST_FROM_START_PLACEHOLDER:
                        dn["dist_from_start_global"] = n["dist_from_start_global"] + 1 
                        dn["dist_from_start_originator"] = dist_from_start_originator
                        next_nodes.append(dn)
            return next_nodes
        
        input_nodes_global = get_input_children(self.nodes)
        print("num global input nodes", len(input_nodes_global))

        # for o in input_nodes_global: TODO revisit if want to use this instead
        #     o["total_downstream_nodes"] = len(get_all_up_or_downstream_nodes(o, self.nodes, get_downstream_ops))
        # # input_nodes_global.sort(key=lambda x: x["total_downstream_nodes"], reverse=True) 
        # input w most downstream ops is primary
        
        input_nodes_global.sort(key=lambda x: x["dist_from_end_global"]) 
        # NOTE primary decision rule here. When considering this, also consider rule below for modules

        for i,n in enumerate(input_nodes_global):
            n["dist_from_start_originator"] = i
            n["dist_from_start_global"] = 0
            next_nodes = mark_and_return_next_unmarked_nodes([n], i)
            while len(next_nodes) > 0:
                next_nodes = mark_and_return_next_unmarked_nodes(next_nodes, i)
            
        # n_nodes_unmarked = len([n for n in self.nodes if n["dist_from_start_global"]==DIST_FROM_START_PLACEHOLDER])
        n_nodes_unmarked = len([n for n in self.nodes_lookup.values() if n["dist_from_start_global"]==DIST_FROM_START_PLACEHOLDER])

        print(f"There are {n_nodes_unmarked} nodes left unmarked for 'dist to start'")
        # print([(n["name"], n["dist_from_start_global"]) for n in self.nodes_lookup.values()])

    def denote_module_children(self):
        ###############################
        # Groups. Flat. Only taking into account a group and its children (ie not their subchildren)
        # this will be nested below
        # This now nests them automatically

        for module in self.modules:
            module_name_id = module["name"]+"-"+module["node_id"]

            nodes_w_parents = [n for n in self.nodes if len(n["parent_ops"])>0] #SD has some nodes w no parents?
            # children_nodes = [n for n in self.nodes if n["parent_ops"][-1]==module_name_id]
            children_nodes = [n for n in nodes_w_parents if n["parent_ops"][-1]==module_name_id]
            non_root_modules = [m for m in self.modules if len(m["parent_ops"])>0]
            children_modules = [mod for mod in non_root_modules if mod["parent_ops"][-1]==module_name_id]
            
            module["children"] = []
            for n in children_nodes:
                n.update({
                    "children":[], # for convenience, will always be empty
                    "collapsed":True, # for convience. Will always be true
                    "h":0,
                    "w":0,
                    "n_ops":1,
                    "history":[]
                })
                module["children"].append(n)

            for m in children_modules:
                m.update({
                    "collapsed": False, # toggle in JS. Default expanded
                    "n_ops":0,
                    "history":[],
                    "h":0, # init at zero, will be expanded
                    "w":0,
                })
                module["children"].append(m)

    def nestify(self):
        # now a misnomer, as nesting happens automatically above. Used to create copies, then had to manually nest them
        # but we're now nesting them in place, so can refer to flat version or nested version and all works

        root_op = [g for g in self.modules if len(g["parent_ops"])==0] #TODO we know the name of this, we created it. Use that instead
        assert len(root_op)==1, "must be only one root op"
        hierarchy = root_op[0]

        ###############################
        # mark n ops
        def mark_n_ops(op):
            for c in op["children"]:
                mark_n_ops(c)
                op["n_ops"] += c["n_ops"]

        hierarchy["n_ops"] = 0
        mark_n_ops(hierarchy)

        ###############################
        # Mark depth
        def mark_depth(op):
            for c in op["children"]:
                c["depth"] = op["depth"]+1
                mark_depth(c)
        hierarchy["depth"] = 0
        mark_depth(hierarchy)

        # ###################################
        # # marking io 

        # # doing bc modules don't yet have io markings TODO shouldn't need this
        module_is_input = 0; module_is_output = 0
        def mark_io(op):
            nonlocal module_is_input, module_is_output
            input_children = get_input_children(op["children"])
            for c in input_children: 
                if c["node_type"]=="module": 
                    module_is_input += 1
                    c["is_input"] = True
            output_children = get_output_children(op["children"])
            for c in output_children: 
                if c["node_type"]=="module":
                    module_is_output += 1
                    c["is_output"] = True
            for c in op["children"]:mark_io(c)

        mark_io(hierarchy)
        print(f"Module inputs: {module_is_input}. Module outputs: {module_is_output}")


        for n in self.nodes:
            n["respath_dist"] = n["dist_from_start_global"] + n["dist_from_end_global"]

        ###############################
        # dist from end / start global, for modules. Nodes marked above globally
        def mark_dist_from_start_and_end(op):
            for c in op["children"]: mark_dist_from_start_and_end(c)

            input_children = [c for c in op["children"] if c.get("is_input")]

            # TODO modules should no longer need these, as we're determining them based on the children io nodes
            if op.get("respath_dist") is None: # only true from modules. Nodes marked above. 
                # TODO Actually needs to take into account the specific input node... 
                op["respath_dist"] = min([c["respath_dist"] for c in input_children]) \
                                                        if len(input_children)>0 else 1e6
            
            # if op.get("dist_from_start_global") is None:
            #     op["dist_from_start_global"] = min([c["dist_from_start_global"] for c in input_children]) \
            #                                             if len(input_children)>0 else 1e6
                            
        mark_dist_from_start_and_end(hierarchy)


        ###################################
        # propogate input ix to modules, marked for nodes above
        def mark_input_originator(op):
            if len(op["children"])>0:
                for c in op["children"]: mark_input_originator(c)
                # if a module has no children, will get KeyError here for dist_from_start_originator
                op["dist_from_start_originator"] = min([o["dist_from_start_originator"] for o in op["children"]])
                # module takes the most primary of its subops. If if has a primary and a conditioning, takes primary
        
        mark_input_originator(hierarchy)


        self.timer.log("nestify first stuff")

        ###############################
        # X pos. Relative.

        def mark_next_x_pos(op, peers):
            downstream_ops = get_downstream_ops(op, peers)
            for d in downstream_ops:
                v = op["x_relative"] + 1
                if v > d["x_relative"]:
                    d["x_relative"] = v
                    mark_next_x_pos(d, peers)

        P = 1e6
        def mark_children_x(children):
            if len(children)==0: return 

            for c in children: c["x_relative"] = -P

            input_ops = [c for c in children if c.get("is_input")]
            for i,input_op in enumerate(input_ops):
                # if input_op.get("x_relative") == -P: # modules sometimes already marked. Don't set those to zero
                input_op["x_relative"] = 0
                mark_next_x_pos(input_op, children)

            # # normalize to start at zero TODO don't need this?
            # children_min_x = min([c["x_relative"] for c in children])
            # for c in children: c["x_relative"] -= children_min_x

            ##### move io nodes to front and back of module
            children_max_x = max([c["x_relative"] for c in children])

            for c in children:
                # if c.get("is_input"): c["x_relative"] = 0 # TODO is always zero 
                if c.get("is_output"): c["x_relative"] = children_max_x # TODO only if returned by module
                # remember this acts within the module. Once things are expanded, the output nodes won't
                # necessarily be at the end. But they will be at the end for all calculations within the mod,
                # ie when determining Y height, so they'll be there blocking as we need them to be

            ##############
            for c in children: mark_children_x(c["children"])

        hierarchy["x_relative"] = 0
        mark_children_x(hierarchy["children"])

        unmarked = []
        def check_unmarked(op):
            if op.get("x_relative")==None:
                unmarked.append(op)
            elif op.get("x_relative") > abs(P/2):
                unmarked.append(op)
            for c in op["children"]: check_unmarked(c)
        check_unmarked(hierarchy)

        print("n unmarked x_relative", len(unmarked))
        if len(unmarked)>0: print([o["name"] for o in unmarked])
        for u in unmarked: 
            u["x_relative_was_left_unmarked"] = True #TODO debugging, remove
            u["x_relative"] = 0

        self.timer.log("x pos")


        ############################################
        # Extension lines
        # def get_dist_from_end(op, upstream_op):
        #     if op["node_type"]=="module":
        #         input_node = [c for c in op["children"] if intx(c["inbound_edges"], upstream_op["outbound_edges"])]
        #         # TODO dla breaks the assert, tracer needs to combine
        #         input_node = input_node[0]
        #         assert input_node.get("is_input")==True
        #         return input_node["dist_from_end_global"]
        #     else:
        #         return op["dist_from_end_global"]
        def get_respath_dist(op, upstream_op):
            if op["node_type"]=="module":
                input_node = [c for c in op["children"] if intx(c["inbound_edges"], upstream_op["outbound_edges"])]
                # TODO dla breaks the assert, tracer needs to combine
                input_node = input_node[0]
                assert input_node.get("is_input")==True
                return input_node["respath_dist"]
            else:
                return op["respath_dist"]
            
        n_extension_lines = 0; n_extension_nodes = 0

        def add_extension_line(root_base_node, all_dns, extension_nodes_container, n_dns, double_elbow):
            nonlocal n_extension_lines
            n_extension_lines += 1
            assert len(root_base_node["outbound_edges"])==1 # only calling this fn when only one outbound tensor
            original_edge = root_base_node["outbound_edges"][0] # always just one. Same one being passed forward. 
            root_base_node["is_extension_line_base"] = True
            tensor_is_used_many_times = n_dns >= 3
            root_base_node["tensor_is_used_many_times"] = tensor_is_used_many_times
            node_type = "elbow" if n_dns==1 else "extension"

            def _extend_line(base_node, dns):
                nonlocal n_extension_nodes
                n_extension_nodes += 1
                # if elbow, use from dn. This matters bc if have one elbow and one not, then elbow has same as 
                # base but other is one less, so always take other. Affects coat mini. 
                # dist_end = get_respath_dist(all_dns[0], base_node) if n_dns==1 else base_node["dist_from_end_global"]
                
                dns.sort(key=lambda x : x["x_relative"])
                branch_node = {} #copy.deepcopy(base_node)

                closest_x = dns[0]["x_relative"]
                branch_node["x_relative"] = closest_x - 1
                
                respath_dist = get_respath_dist(dns[0], base_node) if n_dns==1 else base_node["respath_dist"] 
                # TODO hack, this will be deleted soon anyways

                branch_node["n_ops"] = 0
                branch_node["respath_dist"] = respath_dist
                branch_node["node_type"] = node_type
                branch_node["children"] = []
                branch_node["history"] = []
                branch_node["collapsed"] = True
                branch_node["name"] = node_type
                branch_node["node_id"] = base_node["node_id"]+"_ext"
                branch_node["tensor_is_used_many_times"] = tensor_is_used_many_times


                new_edge_in = original_edge+"_ext_"+str(closest_x)+"_in"
                base_node["outbound_edges"] += [new_edge_in]

                # this can happen on the first base node TODO shouldn't happen anymore bc using output tensors
                if base_node["node_type"]=="module":
                    base_node_outputs = [n for n in base_node["children"] if n.get("is_output")]
                    for o in base_node_outputs:
                        if original_edge in o["outbound_edges"]:
                            o["outbound_edges"] += [new_edge_in]
                            o["is_extension_line_base"] = True 
                            o["tensor_is_used_many_times"] = tensor_is_used_many_times
                            # TODO awkward. relying on fact that this is always first base node, 
                            # ie not further into ext line
                            branch_node["respath_dist"] = o["respath_dist"] # otherwise takes module's, coat mini
                
                branch_node["inbound_edges"] = [new_edge_in]

                new_edge_out = original_edge+"_ext_"+str(closest_x)+"_out"
                branch_node["outbound_edges"] = [new_edge_out]

                rerouted_dns = [dn for dn in dns if dn["x_relative"]==closest_x]
                branch_node["dn_ids"] = [n["node_id"] for n in rerouted_dns] # used in JS to keep them one less than dn after expansions
                for dn in rerouted_dns:
                    dn["inbound_edges"] = [e for e in dn["inbound_edges"] if e != original_edge]
                    dn["inbound_edges"].append(new_edge_out)

                    # if dn is a module, also have to connect to its input nodes. 
                    # this will never be nested bc all inputs go through input nodes. Only have to connect inside one 
                    # nested level
                    if dn["node_type"]=="module":
                        dn_inputs = [n for n in dn["children"] if n.get("is_input")]
                        for dn_input in dn_inputs:
                            if original_edge in dn_input["inbound_edges"]: # should only be one
                                dn_input["inbound_edges"] = [e for e in dn_input["inbound_edges"] if e != original_edge]
                                dn_input["inbound_edges"].append(new_edge_out) 

                extension_nodes_container.append(branch_node) # adding these in-place

                candidate_dns = [dn for dn in dns if dn["x_relative"]>(branch_node["x_relative"]+1)]
                if len(candidate_dns)>=1: _extend_line(branch_node, candidate_dns) # once started, continue all the way
            
            _extend_line(root_base_node, all_dns)


        def consolidate_edges_if_necessary(base_node, peer_ops, extension_nodes_container):
            base_node["checked_for_edge_consolidation"] = True
            dns = get_downstream_ops(base_node, peer_ops)


            if base_node.get("x_relative") is not None:
                candidate_dns = []
                for dn in dns:
                    if dn.get("x_relative") is not None:
                        if dn.get("x_relative")>(base_node.get("x_relative")+1):
                            candidate_dns.append(dn)
                
                double_elbow = False
                if len(dns)>1 and len(candidate_dns)==1:
                    double_elbow = True
                if len(base_node["outbound_edges"])==1 and len(candidate_dns)>=1: # don't start unless >= n
                    add_extension_line(base_node, candidate_dns, extension_nodes_container, len(candidate_dns), double_elbow)
                
            for dn in dns:
                if not dn["checked_for_edge_consolidation"]:
                    consolidate_edges_if_necessary(dn, peer_ops, extension_nodes_container)


        def consolidate_edges(peer_ops):
            for op in peer_ops: op["checked_for_edge_consolidation"] = False

            input_ops = [op for op in peer_ops if op.get("is_input")]
            # input_ops.sort(key=lambda x: x["input_priority"]) # TODO why is this necessary?
            extension_nodes_container = [] # container to be added to inplace
            for op in input_ops:
                consolidate_edges_if_necessary(op, peer_ops, extension_nodes_container)

            ############
            for op in peer_ops: consolidate_edges(op["children"])

            peer_ops += extension_nodes_container # ensure this does in-place relative to peer_ops
            self.nodes += extension_nodes_container # bc we're now also keeping flat version to work w, currently in conditioning below
            # TODO this is awkward, having to keep flat and nested up to date manually. Need better data structure here. 

        consolidate_edges(hierarchy["children"])

        print(f"added {n_extension_lines} extension lines and {n_extension_nodes} extension nodes")

        ###############################
        # denoting conditioning pathways

        # we can now work w nodes and modules in nested form or flat form, they both represent the same underlying objects.
        # different tasks are easier in different forms.

        # tensors are conditioning, not nodes. But we're hacking together for now just using nodes bc aren't showing
        # tensors explicitly yet. 
        # Conditioning is best defined at the site of interaction, ie two tensors going into an op, one can be conditioning
        # and the other can be 'primary stream'. This is akin to modulatory/mediating interactions in the brain. From that site
        # of modulation, we can trace upstream and say that any computation that leads solely to that conditioning tensor is 
        # also purely conditioning. Conditioning is relative, ie some ops may condition one site, then the whole group together
        # may condition something else. This is clear in SD, where a CLIP model provides conditioning tensors for the unet, but
        # the clip itself has conditioning information within it. 
        # We can find conditioning pathways anywhere along the path, then trace them up and downstream. Up/downstream traces 
        # until there is other computation entering the stream, ie upstream until the tensors are used elsewhere outside conditioning
        # and downstream until the conditioning site itself, defined as where other, nonconditioning, information enters into the 
        # stream. Imagine it hitting the inner corner of an outgoing (upstream) or incoming (downstream) branch to start or end
        # the pathway. 
        #
        # indicate a point somehwere along the conditioning pathway, then trace upstream and downstream to mark the entire
        # pathway. 
        # Conditioning vs primary is important when choosing the respath. At a branch, we choose the path that leads closest
        # to the primary output. Unless it is a conditioning pathway, then even if it's faster we want to take one of the others.
        # Input priority is also determined by respath criteria, ie dist to end, and is affected by conditioning vs primary categorization
         
        n_conditioning = 0
        def mark_conditioning_downstream(n): # only tensors are conditioning, but just hacking together for now
            if not n.get("is_conditioning")==True:
                n["is_conditioning"] = True
                nonlocal n_conditioning
                n_conditioning += 1
                downstream_nodes = get_downstream_ops(n, self.nodes)
                for dn in downstream_nodes:
                    dn_uns = get_upstream_ops(dn, self.nodes)
                    keep_going = all([nn.get("is_conditioning")==True for nn in dn_uns])
                    if keep_going:
                        mark_conditioning_downstream(dn)

        def mark_conditioning_upstream(n): # only tensors are conditioning, but just hacking together for now
            if not n.get("is_conditioning_upstream")==True:
                n["is_conditioning_upstream"] = True # be careful with these, don't want to mark directly above if not appropriate
                nonlocal n_conditioning
                n_conditioning += 1
                upstream_nodes = get_upstream_ops(n, self.nodes)
                for un in upstream_nodes:
                    un_dns = get_downstream_ops(un, self.nodes)
                    keep_going = all([nn.get("is_conditioning_upstream")==True for nn in un_dns])
                    if keep_going:
                        mark_conditioning_upstream(un)

        for n in self.nodes:
            if (n.get("is_extension_line_base")==True and n.get("tensor_is_used_many_times")==True) or \
                    n.get("not_respath")==True: # be careful with these, don't want to mark directly above if not appropriate
                mark_conditioning_downstream(n)
                mark_conditioning_upstream(n)

        # keeping modules up to date. If all subops are conditioning, op is conditioning
        def mark_conditioning_module(op):
            
            for c in op["children"]:
                if len(c["children"])>0:
                    mark_conditioning_module(c)
            
            op["is_conditioning"] = all([(c.get("is_conditioning_upstream") or c.get("is_conditioning")) 
                                         for c in op["children"]])
        
        
        mark_conditioning_module(hierarchy)
            
        ###################################
        # marking io priority

        def mark_input_priority(op):
            input_children = [c for c in op["children"] if c.get("is_input")]

            input_children.sort(key=lambda x : (
                (1 if (x.get("is_conditioning") or x.get("is_conditioning_upstream")) else 0),
                x["dist_from_start_originator"], # primary input path first
                # x["dist_from_end_global"], 
                # x["dist_from_end_global"]+x["dist_from_start_global"], 
                x["respath_dist"], 
                # x["dist_from_start_global"]
            ))
            
            for i,c in enumerate(input_children): c["input_priority"] = i

            for c in op["children"]: mark_input_priority(c)

        mark_input_priority(hierarchy)

        print(f"marking {n_conditioning} nodes as conditioning")
        ###############################
        # Y pos. Relative.

        def respath_sort(arr, upstream_op):
            arr.sort(key=lambda x: ((0 if x["node_type"]=="extension" else 1), # extensions always extend
                                    (1 if (x.get("is_conditioning") or x.get("is_conditioning_upstream")) else 0),
                                    # x['dist_from_end_global']+x["dist_from_start_global"]
                                    get_respath_dist(x, upstream_op)
                                    # x['dist_from_end_global']
                                    ))  # Sorting by respath criteria

        def mark_y(start_nodes, nodes, occupancy, counter):
            # for n in nodes: n["y_relative"] = None
            # only nodes w (y_relative == None) should enter here.

            def mark_next_y(nodes_arr):
                next_nodes = [] 
                    
                def get_flat_nodes_line(node, collect_branch_nodes=True):
                    nodes_line = []
                    terminates_at_x, terminates_at_y = node["x_relative"], node.get("y_relative") # not included in nodes_line. 

                    def add_to_nodes_line(_node):
                        nodes_line.append(_node)

                        nonlocal terminates_at_x, terminates_at_y, next_nodes

                        if _node["node_type"]=="module": # NOTE this change increased time substantially, i believe
                            downstream_nodes = []
                            module_output_nodes = [_n for _n in _node["children"] if _n.get("is_output")]
                            module_output_nodes.sort(key=lambda x : x["y_relative"])
                            for output_node in module_output_nodes:
                                dns = get_downstream_ops(output_node, nodes) # there will only be one of these, the mod out
                                respath_sort(dns, _node)
                                downstream_nodes += dns
                        else:
                            downstream_nodes = get_downstream_ops(_node, nodes)
                            if len(downstream_nodes)==0: return
                            respath_sort(downstream_nodes, _node)

                        # Get branch nodes out of the way
                        # when just measuring x_len, don't push more branch nodes to next_nodes
                        if collect_branch_nodes:
                            branch_nodes = [bn for bn in downstream_nodes[1:] if bn.get('y_relative') is None]
                            next_nodes += branch_nodes

                        # Continue along respath
                        respath_dn = downstream_nodes[0]  # Following respath
                        terminates_at_x, terminates_at_y = respath_dn["x_relative"], respath_dn.get("y_relative")

                        if respath_dn.get('y_relative') is None: add_to_nodes_line(respath_dn)

                    add_to_nodes_line(node)

                    return nodes_line, (terminates_at_x, terminates_at_y)
                
                # Choose next node for starting a flat line. These will all be branch nodes. 
                # Except for start nodes but those are passed in one at a time
                # and except for when stop early bc of output node that still has other dns within module
                for n in nodes_arr:
                    _, _terminates_at = get_flat_nodes_line(n, collect_branch_nodes=False)
                    n["terminates_at_x"] = _terminates_at[0] 
                    n["terminates_at_y"] = _terminates_at[1]

                nodes_arr.sort(key=lambda x : (-x["x_relative"], x["terminates_at_x"])) 
                # closer to end first, then the ones that end first go first. 
                # NOTE this is one of the few primary rules we're using, give it great care

                base_node = nodes_arr[0] # this will start the next flat line
                next_nodes += nodes_arr[1:] # pass on remaining nodes for next time

                # get nodes line
                nodes_line, terminates_at = get_flat_nodes_line(base_node)

                last_node = nodes_line[-1]
                
                max_x = last_node["x_relative"] # max x of line itself
                block_until = max_x+1

                block_from = base_node["x_relative"]

                # TODO undo this [-1]
                occ_y = max(occupancy[base_node['x_relative']: block_until]+[-1]) # error here where x_relative is unmarked, bc not int
                row_height = occ_y+1

                # #############################
                # # Deal w overlap btwn last node in row and its downstream nodes

                # downstream_nodes = get_downstream_ops(last_node, nodes)
                # downstream_nodes = [n for n in downstream_nodes if n.get("y_relative") is not None]
                # downstream_nodes.sort(key=lambda x: x["y_relative"])

                # row_moved_up, extra_blocked = [], False
                # # Move up row if creates horizontal line to dn and there is blocking occupancy in btwn
                # # dn must already be marked in this case
                # #TODO i believe can delete
                # for dn in downstream_nodes:
                #     occ_y_from_end_of_line_to_dn = max(occupancy[block_until:dn["x_relative"]]+[-1])
                #     if (row_height==dn["y_relative"]) and (row_height<=occ_y_from_end_of_line_to_dn):
                #         row_height = occ_y_from_end_of_line_to_dn+1
                #         row_moved_up.append(f"row moved up by dn {dn['name']} bc in btwn occ blocked: y occ measured at {occ_y_from_end_of_line_to_dn} from x {block_until} to {dn['x_relative']}. Last node in row is {nodes_line[-1]['name']} whose x is {nodes_line[-1]['x_relative']}")
                #         dn["history"].append(f"caused row ending in {nodes_line[-1]['name']} to move upwards bc occ blocked in btwn")

                # # If connecting to an upstream node creates a horizontal line, block occupancy
                # # TODO prob have to do the same thing as above, ie "if creates line and has blockage, move up, ",
                # # as well as this just-blockage
                # # TODO when is this used?
        
                # upstream_nodes = get_upstream_ops(base_node, nodes)
                # upstream_nodes = [n for n in upstream_nodes if n.get("y_relative") is not None]
                # upstream_nodes.sort(key=lambda x: x["y_relative"])
                # for un in upstream_nodes:
                #     if (row_height==un["y_relative"]):            
                #         block_from = un["x_relative"]
                #         assert block_from < base_node["x_relative"]


                # # If last node of row to dn creates a straight horizontal line 
                # # (but nothing blocking in btwn bc we just dealt w that)
                # # then leave row in place but block occupancy up to the dn
                # for dn in downstream_nodes:
                #     if (row_height==dn["y_relative"]):            
                #         block_until = dn["x_relative"]
                #         extra_blocked = True

                # ##############################

                # mark nodes line
                nonlocal counter
                for n in nodes_line: 
                    n["y_relative"] = row_height
                    n["draw_order"] = counter[0]
                    n["row_counter"] = counter[1]
                    n["terminates_at_x"] = terminates_at[0]
                    n["terminates_at_y"] = terminates_at[1]
                    # n["history"] += row_moved_up
                    # if extra_blocked: n["history"].append("occ after row blocked")
                    counter[0] += 1
                counter[1] += 1

                # update occupancy
                for i in range(block_from, block_until): occupancy[i] = max(row_height, occupancy[i])

                return next_nodes

            next_nodes = mark_next_y(start_nodes)
            c = 0
            while (len(next_nodes) > 0):
                c += 1
                if (c > 1000): print("too many y iters")
                next_nodes = mark_next_y(next_nodes)

        
        # each group is still being marked relatively, ie not nested, though we're traversing in a nested way
        # Doing this after nesting bc need the dist_from_end_global we get only after nesting. 
        # To reiterate: no interaction btwn the nested levels.
        def mark_children_y(children):
            if (len(children)>0):
                # Ensure all children are marked before marking this one. Parent will use their information.
                for c in children: mark_children_y(c["children"])

                # doing each child separately bc want to first sort based on input criteria
                # so passing in a child, doing all downstream, then passing in next child. 
                # Passing occupancy through bc all use same one of course. Used to pass in all children side by side,
                # but then sorting inside the fn seemed challenging. This gives desired behavior: choose input based
                # on dist to end, then do all downstream using our normal rules, then choose next input based on dist to end, etc
                occupancy = [-1 for _ in range(int(1e6))]
                counter = [0, 0] # so object itself updates as passed through, same as occupancy
                input_children = [c for c in children if c.get("is_input")]
                # input_children.sort(key=lambda x : x["dist_from_end_global"]) # respath criteria
                # TODO sort inputs in same way for X and Y
                                                             
                input_children.sort(key=lambda x : x["input_priority"]) 
                # don't know about this. Might want normal respath criteria
                for c in input_children:
                    if (c.get("y_relative") is None): # if input node gets already marked in a first pass, don't send through again.
                        mark_y([c], children, occupancy, counter)

        mark_children_y(hierarchy["children"])

        self.timer.log("y pos")


        self.nested_hierarchy = hierarchy
