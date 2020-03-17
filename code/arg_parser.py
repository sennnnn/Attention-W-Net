import os
import sys

from util import dict_load

class arg_parser(object):
    args_dict = {}
    key_list = []
    map_dict = \
    {

    }
    special_key_list = []
    def __init__(self):
        self.special_key_list += list(self.map_dict.keys())
        self.sys_args = sys.argv[1:]

    def add_map(self, arg_key, arg_map_dict):
        assert '--' in arg_key, 'Error, the arg key must be the parttern like --xxx'
        if(arg_key in self.special_key_list):
            print('The arg {} has been created.'.format(arg_key))
        else:
            self.map_dict[arg_key] = arg_map_dict
            self.special_key_list.append(arg_key)

    def add_val(self, arg_key, arg_default_val):
        """
        insert item to parse list.
        Args:
            arg_key: the arg which will be the key, and it must seem like --xxx.
            arg_default_val: give a prior val to key.
        Return:
            None
        """
        assert '--' in arg_key, 'Error, the arg key must be the parttern like --xxx'
        self.args_dict[arg_key] = arg_default_val
        self.key_list.append(arg_key)

    def _convert(self, val, val_type):
        if(val_type == type(1)):
            val = int(val)
        if(val_type == type(1.1)):
            val = float(val)
        if(val_type == type(True)):
            val = True if(val == 'True') else False if(val == 'False') else None
        
        return val

    def __call__(self):
        """
        Extract the arg like --xxx as the key, and the next arg is the val.
        Args:
            sys_args: raw command line args.
        Return:
            args_dict: like {'--abc': 'abc'}
        """
        sys_args = self.sys_args
        i = 0
        while(i < len(sys_args)):
            arg = sys_args[i]
            i += 1
            if(arg in self.special_key_list):
                val = sys_args[i]
                self.args_dict[arg] = self.map_dict[arg][val]
                i += 1
                continue
            if(arg in self.key_list):
                val = sys_args[i]
                val = self._convert(val, type(self.args_dict[arg]))
                assert val != None, 'Error, arg {} get invalid type input.'.format(arg)
                self.args_dict[arg] = val
                i += 1
                continue

        return self.args_dict

def args_process():

    ret_dict = {}

    a = arg_parser()

    a.add_val('--task', 'train')

    model_dict = {'unet':'unet', 'r2u':'r2Unet', 'att':'attentionUnet', 'hyb':'hybridUnet', 'uplus':'unetPlusPlus'}

    model_pattern_dict = {'ckpt':'ckpt', 'pb':'pb'}

    target_dict = {'HN_OAR': 'HaN_OAR', 'Lu_OAR': 'Thoracic_OAR', 'HN_GTV': 'Naso_GTV', 'Lu_GTV': 'Lung_GTV'}

    a.add_map('--model', model_dict)
    a.add_map('--pattern', model_pattern_dict)
    a.add_map('--target', target_dict)

    parse_dict = a()

    training_metric_loss_only_log_path = os.path.join('build/{}-{}'.format(parse_dict['--model'], \
                                         parse_dict['--target']), 'valid_metric_loss_only.log')
    start_epoch = 1
    if(not os.path.exists(training_metric_loss_only_log_path)):
        last = False
    else:
        try:
            temp_dict = dict_load(training_metric_loss_only_log_path)
            start_epoch = len(temp_dict['epochwise']['metric']) + 1
            if(start_epoch == 0):
                last = False
            else:
                last = True
        except:
            last = False

    ret_dict['task'] = parse_dict['--task']
    ret_dict['model'] = parse_dict['--model']
    ret_dict['model_pattern'] = parse_dict['--pattern'] if('--pattern' in parse_dict.keys()) else None
    ret_dict['target'] = parse_dict['--target']
    ret_dict['last'] = last
    ret_dict['start_epoch'] = start_epoch

    return ret_dict