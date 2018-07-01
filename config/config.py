# __author__=   'Sargam Modak'

import yaml


def load_config(yamlfile_path='./config/config.yaml'):
	with open(yamlfile_path) as yaml_file:
		cfg = yaml.load(yaml_file)
	return cfg
