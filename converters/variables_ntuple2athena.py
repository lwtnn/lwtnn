#!/usr/bin/env python3
#
# Converter from NN configuration with variables used in the training ntuple to the convention used in Athena
"""
____________________________________________________________________
NN configuration file

This is the output of one of the converters, e.g. keras2json.py.
"""

import argparse
import json

def _run():
    """Top level routine"""
    args = _get_args()
    with open(args.config_file, 'r') as config_file:
        nn_config = json.load(config_file)

    nn_config = _update_naming_convention(nn_config)

    with open(args.config_file.replace('.json', '_athena-conventions.json'), 'w') as output_file:
        json.dump(nn_config, output_file, indent=2)
    print("Dumped NN configuration file with variable convention as used in Athena to {}.".format(args.config_file.replace('.json', '_athena-conventions.json')))

def _get_args():
    parser = argparse.ArgumentParser(
        description="Converter from variables used in the training ntuple to the convention used in Athena",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config_file', help='NN configuration json file')
    return parser.parse_args()

_variable_convention_converter = {
    'eta_abs_uCalib': 'jet_abs_eta',
    'pt_uCalib': 'jet_pt',
    'jf_mass': 'jf_m',
    'jf_mass_check': 'jf_m_check',
    'jf_efrc': 'jf_efc',
    'jf_sig3': 'jf_sig3d',
    'jf_ntrkv': 'jf_ntrkAtVx',
    'jf_n2tv': 'jf_n2t',
    'jf_dR': 'jf_dRFlightDir',
    'sv1_ntkv': 'sv1_ntrkv',
    'sv1_mass': 'sv1_m',
    'sv1_mass_check': 'sv1_m_check',
    'sv1_efrc': 'sv1_efc',
    'sv1_sig3': 'sv1_sig3d',
    }

_output_labeling_convention_converter = {
    'b-jet': 'bottom',
    'c-jet': 'charm',
    'u-jet': 'light'
}

def _update_naming_convention(network):
    """Update naming convention from the one used during training to the one used in Athena"""
    input_variables_list = _get_input_variables(network)
    output_variables_list = _get_output_variables(network)

    # replace 'default' key string with the one used in Athena:
    for variable_name in input_variables_list:
        if variable_name in list(network['defaults'].keys()):
            if variable_name in list(_variable_convention_converter.keys()):
                network['defaults'][_variable_convention_converter.get(variable_name)] = network['defaults'].get(variable_name)
                del network['defaults'][variable_name]

    # replace 'input' 'name' string with the one used in Athena:
    for variable_itr, variable_item in enumerate(network.get('inputs')):
        if variable_item.get('name') in list(_variable_convention_converter.keys()):
            network['inputs'][variable_itr]['name'] = _variable_convention_converter.get(variable_item.get('name'))

    # replace 'output' label string with the one used in Athena:
    for variable_itr, variable_name in enumerate(output_variables_list):
        if variable_name in list(_output_labeling_convention_converter.keys()):
            network['outputs'][variable_itr] = _output_labeling_convention_converter.get(variable_name)

    return network

def _get_input_variables(network):
    input_variables = []
    for variable in network.get('inputs'):
        input_variables.append(variable.get('name'))
    return input_variables

def _get_output_variables(network):
    output_variables = []
    for variable in network.get('outputs'):
        output_variables.append(variable)
    return output_variables

if __name__ == '__main__':
    _run()
