from os import path
from typing import Dict, List


def extract_path_from_config(conf, inic_path: str) -> List[str]:
	RESOLUCAO = f"{conf['size']}x{conf['size']}"
	PARTICIONAMENTO = conf['partic']

	EQ_HIST = 'hist-equal' if conf.get('equal_hist', False) else ''
	TRANSF_MORF = 'transf-morf' if conf.get('transf_morf', False) else ''

	COLORIDO = 'cor' if conf.get('cor', False) else ''
	COM_CAUSA = 'causa' if conf.get('causa', False) else ''

	DATASET_CONFIG = '_'.join([
		opcao for opcao in [RESOLUCAO, PARTICIONAMENTO, EQ_HIST, TRANSF_MORF, COLORIDO, COM_CAUSA]
		if opcao
	])

	return [
		path.join(p, RESOLUCAO, f'custom_{DATASET_CONFIG}')
		for p in inic_path
	]


def build_config_with_paths(conf) -> Dict:
	sufixo = '_gt'
	path_nome_final: Dict[str, str] = {
		'train_x_dirs': 'train',
		'train_y_dirs': f'train{sufixo}',
		'valid_x_dirs': 'val',
		'valid_y_dirs': f'val{sufixo}',
		'test_x_dirs': 'test',
	}
	for nome_var in path_nome_final:
		conf[nome_var] = [
			path.join(p, path_nome_final[nome_var]) for p in
			extract_path_from_config(conf, conf[nome_var])
		]

	return conf

def get_model_file_name(conf) -> str:
	lr = conf["lr"]
	batch_size = conf["batch_size"]
	n_epochs = conf["n_epochs"]
	loss = 'categorical_crossentropy' if conf["loss"] == 'CE' else 'focal_loss'
	optimizer = conf["optimizer"]
	shuffle_train = False
	dropout = conf["dropout"]

	resolucao = f"{conf['size']}x{conf['size']}"
	particionamento = conf['partic']
	eq_hist = 'hist-equal' if conf.get('equal_hist', False) else ''
	transf_morf = 'transf-morf' if conf.get('transf_morf', False) else ''

	dataset_config = '_'.join([
		opcao for opcao in [resolucao, particionamento, eq_hist, transf_morf]
		if opcao
	])

	return f"{dataset_config}_{optimizer.lower()}_{loss}_batch-{batch_size}_epoch-{n_epochs}_LR-{lr}_dropout-{1 if dropout > 0 else 0}_shuffle-{1 if shuffle_train else 0}.h5"

