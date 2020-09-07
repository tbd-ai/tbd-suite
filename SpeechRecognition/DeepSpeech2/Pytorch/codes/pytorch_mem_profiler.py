import torch
from datetime import datetime
import os

OUTPUT_DIR="./memory_csv_data/"

class memory_profiler:
    def __init__(self,model,print_period=1,csv=False):
        """
        Arguments:
            model : torch.nn.Module
                The PyTorch model being trained

            print_period : Integer >=1
                Indicates the number of iterations between reporting
                of memory diagnostics. For example, print_period = 5 
                would report every 5 iterations.

            csv : Boolean
                Indicates whether to report memory diagnostics to 
                a .csv file. Filenames are uniquified by datetime,
                and all values are in MB by default.        
        """
        
        if print_period<1:
            print("ERROR: Please use a print period larger than zero!")
            exit()

        self.csv=csv # boolean for csv outputting

        self.model=model
        self.activation_data_pointers=set()
        self.memory_used_by_feature_maps=0
        self.gradient_data_pointers=set()
        self.memory_used_by_gradients=0

        # Print stats every period of iterations
        self.print_period=print_period
        self.iteration=0
        self.epoch=1

        # Gather the named parameters of the model (i.e. layers)
        self.__gather_named_parameters()

        # Register hooks for feature maps:
        for name, layer in self.model._modules.items():
            layer.register_forward_hook(self.__forward_hook)
            # In case there are submodules, register them as well
            self.__recursive_hooks(layer)

        # Register hooks for gradients
        self.model.register_backward_hook(self.__backward_hook)

        # Initialize the output directory
        if self.csv:
            try:
                if not os.path.exists(OUTPUT_DIR):
                    os.mkdir(OUTPUT_DIR)
            except OSError:
                print(f"Creation of the output directory {OUTPUT_DIR} failed")

            # .csv column labels
            s="epoch,"
            s+="iteration,"
            s+="peak_cached,"
            s+="current_cached,"
            s+="total_feature_map_usage,"
            s+="total_weight_usage,"
            for dp in self.params:
                s+=self.params[dp]["name"]+","                
            s+="total_layer_weight_gradient_usage,"
            for dp in self.params:
                s+=self.params[dp]["name"] + "_grad,"
            s+="intermediate_gradients\n"
            self.fname=OUTPUT_DIR + str(datetime.now().strftime("%Y%m%d%H%M%S")) + ".csv"
            with open(self.fname,"w") as file:
                file.write(s)
            print(f"Logging data in {self.fname}")
        

    def __gather_named_parameters(self):
        """
        Gathers named_parameters from the model.
        """
        # TODO: look into recursively registering submodules
        self.params={} # dp -> {"tensor","size","name","grad_size"}
        for name,param in self.model.named_parameters():
            dp=getDataPtr(param)
            self.params[dp]={}
            self.params[dp]["tensor"]=param # The actual tensor
            self.params[dp]["size"]=getTensorSize(param,scale="B")
            self.params[dp]["name"]=name # user-specified name
            self.params[dp]["grad_size"]=0 # mem of gradient (.grad)


    def __total_layer_mem_MB(self):
        """
        Calculates the total memory usage of all the named 
        parameters in self.params (weights)
        """
        r=0
        for dp in self.params:
            r+=self.params[dp]["size"]
        return MB(r)


    def __recursive_hooks(self, layer):
        '''
        Recursively register forward hooks in all submodules
        of the model. This ensures every activation is accounted
        for during the forward pass.
        '''
        for name, layer in layer._modules.items():
            layer.register_forward_hook(self.__forward_hook)
            self.__recursive_hooks(layer)


    def __forward_hook(self,m, i, o):
        '''
        The hook function to be registered on each module

        Arguments:
            m : torch.nn.Module
            i : tuple of input activation tensors
            o : output activation tensor
        '''

        def expand_list(x):
            if type(x) is torch.Tensor:
                return [x,]
            ret = []
            for x1 in x:
                if type(x1) is torch.Tensor:
                    ret.append(x1)
                elif type(x1) is dict:
                    ret.extend(expand_list(list(x1.values())))
                elif type(x1) is list or type(x1) is tuple:
                    ret.extend(expand_list(list(x1)))

            return ret

        lst = []
        if type(o) is not torch.Tensor:
            if type(o) is dict:
                tmp = expand_list(list(o.values()))
            elif type(o) is list or type(o) is tuple:
                tmp = expand_list(list(o))

            lst.extend(tmp)
        else:
            lst.append(o)

        for o1 in lst:
            if getDataPtr(o1) not in self.activation_data_pointers:
                self.activation_data_pointers.add(getDataPtr(o1))
                self.memory_used_by_feature_maps+=getTensorSize(o1,scale="B")
        
        lst2 = expand_list(i)
        for input_t in lst2:
            if getDataPtr(input_t) not in self.activation_data_pointers:
                self.activation_data_pointers.add(getDataPtr(input_t))
                self.memory_used_by_feature_maps+=getTensorSize(input_t,scale="B")


    def __backward_hook(self, m, in_grads, out_grads):
        """
        Registers hooks for the backward pass. By calling this as:
            model.register_backward_hook(self.__backward_hook)
        all intermediate gradients are registered as well, and not just
        the gradients of leaf nodes in the computational graph.
        https://discuss.pytorch.org/t/using-hook-function-to-save-gradients/4334

        Intermediate gradient tensors are accumulated in C++ buffers, 
        and are only exposed to Python by using hooks.
        https://discuss.pytorch.org/t/how-the-hook-works/2222

        Parameters:
            m : torch.nn.Module
            in_grads: input gradients
            out_grads: output gradients

        """
        # First, inspect the .grad of each module parameter
        for dp in self.params:
            t=self.params[dp]['tensor']
            if t.grad is not None and getDataPtr(t.grad) not in self.gradient_data_pointers:
                self.gradient_data_pointers.add(getDataPtr(t.grad))
                self.memory_used_by_gradients+=getTensorSize(t.grad,scale="B")
                self.params[dp]["grad_size"]+=getTensorSize(t.grad,scale="B")

        # Inspect the input grads to this submodule
        for t in in_grads:
            if getDataPtr(t) not in self.gradient_data_pointers:
                self.gradient_data_pointers.add(getDataPtr(t))
                self.memory_used_by_gradients+=getTensorSize(t,scale="B")

        # Inspect the output grads to this submodule
        for t in out_grads:
            if getDataPtr(t) not in self.gradient_data_pointers:
                self.gradient_data_pointers.add(getDataPtr(t))
                self.memory_used_by_gradients+=getTensorSize(t,scale="B")
        

    def record_stats(self):
        """
        To be called by the user after every iteration. This function
        records statistics about memory usage every print_period
        iterations. If csv logging is enabled, then this also outputs
        memory statistics to a .csv file.

        Note: 
        torch.cuda.max_memory_cached() and 
        torch.cuda.memory_cached() are deprecated in later versions,
        replaced by torch.cuda.max_memory_reserved() and 
        torch.cuda.memory_reserved().
        """
        self.iteration+=1
        
        if self.iteration % self.print_period == 0:
            self.__print_info_table()
            if self.csv:
                self.__write_info_csv()
   
    
    def __print_info_table(self):
        """
        Prints memory diagnostics about weights, gradients, and 
        activations in a user-friendly table. 
        
        The table contains a layer-by-layer breakdown to inform the 
        user about which parts of their model are consuming memory. 
        The layer names are automatically taken from the model 
        supplied by the user. When a user is defining the forward pass,
        there may be some intermediate tensors which are not registered
        as model parameters, and thus no names can be extracted from 
        them. However, their memory is still reported and they will 
        contribute to the total memory in the table.

        When PyTorch is imported, it will immediately consume some 
        memory (on the order of a few hundred MB) which is not 
        attributable to any relevant parts of the model.

        Note: 
        torch.cuda.max_memory_cached() and 
        torch.cuda.memory_cached() are deprecated in later versions,
        replaced by torch.cuda.max_memory_reserved() and 
        torch.cuda.memory_reserved().
        """
        dash = '*' * 43
        print("\n"+dash)
        print("Memory Usage for Iteration", self.iteration, "of Epoch", self.epoch)
        print(dash)

        #print('{:.<35s}{:.>5d} MB'.format("Peak allocated", MB(torch.cuda.max_memory_allocated())))
        #print('{:.<35s}{:.>5d} MB'.format("Current allocated", MB(torch.cuda.memory_allocated())))
        print('{:.<45s}{:.>5d} MB'.format("Peak cached", MB(torch.cuda.max_memory_cached())))
        print('{:.<45s}{:.>5d} MB'.format("Current cached", MB(torch.cuda.memory_cached())))
        print('{:.<45s}{:.>5d} MB'.format("Total feature map usage", MB(self.memory_used_by_feature_maps)))
        
        # Layer-by-layer weight breakdown
        print('\n{:.<45s}{:.>5d} MB'.format("Total layer weight usage", self.__total_layer_mem_MB()))
        for dp in self.params:
            print('{:<2s}{:.<43s}{:.>5d} MB'.format('',self.params[dp]["name"], MB(self.params[dp]["size"])))

        # Total of layer gradients
        total_layer_grad=0
        for dp in self.params:
            total_layer_grad+=self.params[dp]["grad_size"]
        print('\n{:.<45s}{:.>5d} MB'.format("Total layer weight gradient usage", MB(total_layer_grad)))

        # Layer-by-layer gradient breakdown
        self.unnamed_gradient_mem=self.memory_used_by_gradients
        for dp in self.params:
            print('{:<2s}{:.<43s}{:.>5d} MB'.format('',self.params[dp]["name"] + " grad", MB(self.params[dp]["grad_size"])))
            self.unnamed_gradient_mem-=self.params[dp]["grad_size"]

        # Other gradients that are not attributable to specific named layers
        print('{:.<45s}{:.>5d} MB'.format("Intermediate gradients", MB(self.unnamed_gradient_mem)))
    

    def __write_info_csv(self):
        """
        Prints memory diagnostics info to a .csv file. 
        Functionally, this information is identical to
        __print_info_table().

        All values are in MB.

        Note: 
        torch.cuda.max_memory_cached() and 
        torch.cuda.memory_cached() are deprecated in later versions,
        replaced by torch.cuda.max_memory_reserved() and 
        torch.cuda.memory_reserved().
        """
        s=str(self.epoch) + ","
        s+=str(self.iteration) + ","
        #s+=str(MB(torch.cuda.max_memory_allocated()))+","
        s+=str(MB(torch.cuda.max_memory_cached()))+","
        s+=str(MB(torch.cuda.memory_cached()))+","
        s+=str(MB(self.memory_used_by_feature_maps))+","
        s+=str(self.__total_layer_mem_MB())+","
        for dp in self.params:
            s+=str(MB(self.params[dp]["size"]))+","
        total_layer_grad=0
        for dp in self.params:
            total_layer_grad+=self.params[dp]["grad_size"]
        s+=str(MB(total_layer_grad))+","
        for dp in self.params:
            s+=str(MB(self.params[dp]["grad_size"]))+","
        s+=str(MB(self.unnamed_gradient_mem))+"\n"
        with open(self.fname,"a") as file:
            file.write(s)
            
    
    def epoch_end(self):
        '''
        To be called by the user after each epoch is finished.
        '''
        print(f"Epoch {self.epoch} finished")
        self.activation_data_pointers=set()
        self.memory_used_by_feature_maps=0
        self.gradient_data_pointers=set()
        self.memory_used_by_gradients=0
        self.iteration=0
        self.epoch+=1
        for dp in self.params:
            self.params[dp]["grad_size"]=0


def getDataPtr(tensor):
    """
    Get the data pointer of a tensor. The data pointer is used to 
    uniquely identify a tensor in memory. If a reference to the actual
    tensor were stored by the profiler, then PyTorch's reference 
    counting memory management system would not free tensors when
    the model is no longer using them.
    """
    return tensor.storage().data_ptr()


def getTensorSize(tensor, scale="MB"):
    """
    Get the size of a tensor
    """
    element_size = tensor.element_size()
    numel = tensor.storage().size()
    memory_size = numel * element_size
    if scale=="MB":
        return MB(memory_size)
    if scale=="B":
        return memory_size
    print(f"ERROR: unknown scale in getTensorSize: {scale}")
    exit()


def MB(B):
    """
    Convert B bytes to the nearest megabyte.
    """
    return round(float(B)/1000000.0)