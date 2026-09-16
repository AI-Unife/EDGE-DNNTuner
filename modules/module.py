import importlib
from components.colors import colors
import os
import numpy as np
from modules.common_interface import common_interface

class module:
    """
    Class for creating and managing loss module instances
    """
    def __init__(self, modules = []):
        """
        This method initialises the initial attributes in which the informations
        required to manage module instances will be stored:

          - modules_list contains the names of the modules you intend to instantiate,
          passed as argument to the constructor of this class

          - modules_obj contains the list of module instances

          - modules_names contains the names of modules that have been instantiated.
          It's different from the first list, since not all modules may be instantiated

          - modules_ready is a list of booleans, one for each instantiated module,
          indicating whether it conforms to be included in the prolog model

        After the attributes have been initialised, the method to instantiate the modules 'load_modules()' is called 
        """ 
        self.modules_list = modules
        self.modules_obj = []
        self.modules_name = []
        self.modules_ready = []
        self.load_modules()

    def get_module(self, name):
        for i, module in enumerate(self.modules_list):
            if module == name:
                return self.modules_obj[i]

    def load_modules(self):
        """
        Creation of module instances
        """
        for module in self.modules_list:
            # try:
                # each module is stored in the 'loss' folder
                base_dir = "modules.loss." + module

                # with the help of importlib, i try to import the current module
                # if this works, proceed to obtain the class with the same name as the module
                module_class = getattr(importlib.import_module(base_dir), module)

                # each module must use 'common interface' as interface
                # and implement all methods within it
                if issubclass(module_class, common_interface):

                    # if the module is instantiated, put the instance and the name in the dedicated lists.
                    # in addition, set the boolean for inclusion in the prolog model to 'True'
                    self.modules_obj.append(module_class())
                    self.modules_name.append(module)
                    self.modules_ready.append(True)
                else:
                    print(colors.FAIL, f"|  --------- {module} DOESN'T IMPLEMENT INTERFACE  -------  |\n", colors.ENDC)
            # except AttributeError:
            #     print(colors.FAIL, f"|  ----------- {module} CLASS DOESN'T EXIST ----------  |\n", colors.ENDC)
            # except ModuleNotFoundError:
            #     print(colors.FAIL, f"|  ----------- FAILED TO INSTANCIATE {module} ----------  |\n", colors.ENDC)
            #     exit()
            # except NotImplementedError:
            #     print(colors.FAIL, f"|  -------------- ERROR IN {module} STRUCTURE -------------  |\n", colors.ENDC)


    def ready(self):
        """
        Check if at least one loaded module contains no errors for the dynamic creation of the symbolic part
        :return: Boolean indicating if at least one module is suitable for inclusion in the prolog model
        """
        return any(self.modules_ready)
     
    def all_zeros_weights(self):
        """
        Check if all loaded modules have 0 as weights for loss calculation
        :return: Boolean indicating if all loaded modules have 0 as weights for loss calculation
        """
        return (np.sum([module.weight for module in self.modules_obj]) > 0)

    def get_rules(self): 
        """
        Filter rules, actions and problem rules of loaded modules
        :return: Strings containing the set of rules, actions and problem rules of the loaded modules, respectively
        """

        # init rules, actions and problems string to empty strings
        # rules contains prolog rules useful for defining problems that might affect the network
        # actions contains the rules for tuning probabilities
        # problems contains the definition of actions to use given a certain problem
        rules = "\n"
        actions = "\n"
        problems = "\n"

        for index, name in enumerate(self.modules_name):

           # if the module contains no errors for inclusion in the prolog model
           if self.modules_ready[index]:

               # check if the current module defines a symbolic part to be included in the model
               module_name = "modules/loss/" + name + ".pl"
               if os.path.exists(module_name):

                   # comment indicating which rules belong to a module
                   rules += "% rules utils in '" + name + "'\n"
                   actions += "% action rules in '" + name + "'\n"

                   # read the symbolic module part, divide it into lines and iterate over each of them
                   f = open(module_name, 'r')
                   lines = f.readlines()

                   # filter the various lines according to the symbols they contain
                   for line in lines:
                       if "::" in line and ":-" in line:
                           problems += line
                       elif "::" in line:
                           actions += line
                       elif ":-" in line:
                           rules += line
  
                   rules += "\n"
                   actions += "\n"
                   problems += "\n"

                   f.close()

        return rules, actions, problems
    
    def state(self, *args):
        """
        Update internal state of modules
        """
        for index, module in enumerate(self.modules_obj):
            module.update_state(*args)
            
            # if the number of facts defined internally by the module and
            # the number of values returned are different, an error occurs and
            # the model cannot be included in the prolog model.
            # This two lists will be 'zipped' for the creation of the atoms.
            # Different lengths lead to errors during this process
            if len(module.facts) != len(module.obtain_values()):
                self.modules_ready[index] = False

    def values(self):
        """
        Get values of modules
        :return: dict containing {fact name, value} pairs from each module
        """

        # Each module returns a dictionary containing its internal values
        # Initialise the accumulator with an empty dictionary
        values = {}
        for index, (module, name) in enumerate(zip(self.modules_obj, self.modules_name)):

            # Each dict, if the module contains no errors, is accumulated in values dict
            if self.modules_ready[index]:
                values |= module.obtain_values()
            else:
                print(colors.FAIL, f"|  --------- DIFFERENT LENGTHS OF FACTS/VALUES IN {name} --------  |\n", colors.ENDC)

        return values

    def optimiziation(self):
        """
        Calculation of the final value of the loss function
        :return: list of module weights, list of module loss values and final value to be optimised
        """

        # init values and weights accumulators as empty lists
        values = []
        weights = []
        for index, module in enumerate(self.modules_obj):
           if self.modules_ready[index]:

               # accumulate values and weights from each module
               weights += [module.weight]
               values += [module.optimiziation_function()]
               print(f"[DEBUG] Module: {self.modules_name[index]} | Weight: {module.weight} | Value: {values[-1]}")

        # Normalise the values of the weights dividing each of them
        # by the sum of all the accumulated weights
        norm_weights = [w for w in weights]

        # The final value is the sum of the products of the weights by the corresponding values
        final_opt = np.sum([w*v for w,v in zip(norm_weights,values)])

        return weights, values, final_opt

    def print(self):
        """
        Printing module values
        """
        print(colors.OKGREEN, "| MODULE VALUES ---------------------------------------------  |\n", colors.ENDC)
        for index, module in enumerate(self.modules_obj):
            if self.modules_ready[index]:
                module.printing_values()
        print(colors.OKGREEN, "\n | -----------------------------------------------------------  |\n", colors.ENDC)

    def plot(self):
        """
        Plotting graphs of values from each module
        """
        for index, module in enumerate(self.modules_obj):
            if self.modules_ready[index]:
                module.plotting_function()


    def log(self):
        """
        Saving log informations of each module
        """
        for index, module in enumerate(self.modules_obj):
            if self.modules_ready[index]:
                module.log_function()