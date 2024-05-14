from dataset import get_milipoint_dataset

for i in [1, 3, 5]:
	print(f"for stack {i}")
	dataset_config = {
		'seed': 20,
		'train_split': 0.8,
		'val_split': 0.1,
		'test_split': 0.1,
		'stacks': i,
		'zero_padding': 'per_data_point'
	}
	loaders = get_milipoint_dataset(
		'mmr_iden',
		batch_size=32,
		workers=0,
		mmr_dataset_config=dataset_config)
	train_loader, val_loader, test_loader, info = loaders

	for i in train_loader:
		print("Train")
		print(i[0].shape)
		break

	for i in val_loader:
		print("Val")
		print(i[0].shape)
		break

	for i in test_loader:
		print("Test")
		print(i[0].shape)
		break

	print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset), info)
