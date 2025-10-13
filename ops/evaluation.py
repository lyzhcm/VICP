import datetime
import os
import pickle
import numpy as np
import sklearn
import torch
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from torch.nn import functional as F
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional
import tqdm
import glob


class LFold:
	def __init__(self, n_splits=2, shuffle=False):
		self.n_splits = n_splits
		if self.n_splits > 1:
			self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

	def split(self, indices):
		if self.n_splits > 1:
			return self.k_fold.split(indices)
		else:
			return [(indices, indices)]


def calculate_roc(thresholds,
				  embeddings1,
				  embeddings2,
				  actual_issame,
				  nrof_folds=10,
				  pca=0):
	assert (embeddings1.shape[0] == embeddings2.shape[0])
	assert (embeddings1.shape[1] == embeddings2.shape[1])
	nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
	nrof_thresholds = len(thresholds)
	k_fold = LFold(n_splits=nrof_folds, shuffle=False)

	tprs = np.zeros((nrof_folds, nrof_thresholds))
	fprs = np.zeros((nrof_folds, nrof_thresholds))
	accuracy = np.zeros((nrof_folds))
	indices = np.arange(nrof_pairs)

	if pca == 0:
		diff = np.subtract(embeddings1, embeddings2)
		dist = np.sum(np.square(diff), 1)

	for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
		if pca > 0:
			print('doing pca on', fold_idx)
			embed1_train = embeddings1[train_set]
			embed2_train = embeddings2[train_set]
			_embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
			pca_model = PCA(n_components=pca)
			pca_model.fit(_embed_train)
			embed1 = pca_model.transform(embeddings1)
			embed2 = pca_model.transform(embeddings2)
			embed1 = sklearn.preprocessing.normalize(embed1)
			embed2 = sklearn.preprocessing.normalize(embed2)
			diff = np.subtract(embed1, embed2)
			dist = np.sum(np.square(diff), 1)

		# Find the best threshold for the fold
		acc_train = np.zeros((nrof_thresholds))
		for threshold_idx, threshold in enumerate(thresholds):
			_, _, acc_train[threshold_idx] = calculate_accuracy(
				threshold, dist[train_set], actual_issame[train_set])
		best_threshold_index = np.argmax(acc_train)
		for threshold_idx, threshold in enumerate(thresholds):
			tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
				threshold, dist[test_set],
				actual_issame[test_set])
		_, _, accuracy[fold_idx] = calculate_accuracy(
			thresholds[best_threshold_index], dist[test_set],
			actual_issame[test_set])

	tpr = np.mean(tprs, 0)
	fpr = np.mean(fprs, 0)
	from sklearn import metrics
	auc = metrics.auc(fpr, tpr)
	return {
		'fpr': fpr,
		'tpr': tpr,
		'auc': auc,
		'accuracy': accuracy.mean(),
		# 'best_threshold': best_thresholds.mean()
	}


def calculate_accuracy(threshold, dist, actual_issame):
	predict_issame = np.less(dist, threshold)
	tp = np.sum(np.logical_and(predict_issame, actual_issame))
	fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
	tn = np.sum(
		np.logical_and(np.logical_not(predict_issame),
					   np.logical_not(actual_issame)))
	fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

	tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
	fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
	acc = float(tp + tn) / dist.size
	return tpr, fpr, acc


def calculate_val(thresholds,
				  embeddings1,
				  embeddings2,
				  actual_issame,
				  far_target,
				  nrof_folds=10):
	assert (embeddings1.shape[0] == embeddings2.shape[0])
	assert (embeddings1.shape[1] == embeddings2.shape[1])
	nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
	nrof_thresholds = len(thresholds)
	k_fold = LFold(n_splits=nrof_folds, shuffle=False)

	val = np.zeros(nrof_folds)
	far = np.zeros(nrof_folds)

	diff = np.subtract(embeddings1, embeddings2)
	dist = np.sum(np.square(diff), 1)
	indices = np.arange(nrof_pairs)

	for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

		# Find the threshold that gives FAR = far_target
		far_train = np.zeros(nrof_thresholds)
		for threshold_idx, threshold in enumerate(thresholds):
			_, far_train[threshold_idx] = calculate_val_far(
				threshold, dist[train_set], actual_issame[train_set])
		if np.max(far_train) >= far_target:
			f = interpolate.interp1d(far_train, thresholds, kind='slinear')
			threshold = f(far_target)
		else:
			threshold = 0.0

		val[fold_idx], far[fold_idx] = calculate_val_far(
			threshold, dist[test_set], actual_issame[test_set])

	val_mean = np.mean(val)
	far_mean = np.mean(far)
	val_std = np.std(val)
	return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
	predict_issame = np.less(dist, threshold)
	true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
	false_accept = np.sum(
		np.logical_and(predict_issame, np.logical_not(actual_issame)))
	n_same = np.sum(actual_issame)
	n_diff = np.sum(np.logical_not(actual_issame))
	# print(true_accept, false_accept)
	# print(n_same, n_diff)
	val = float(true_accept) / float(n_same)
	far = float(false_accept) / float(n_diff)
	return val, far


def evaluate_verification(model, class_names, root_dir):
	model.eval()
	from ops.dataset import PetFaceEvalDataset
	datasets = []
	for name in class_names:
		import torchvision
		from PIL import Image
		transform = torchvision.transforms.Compose([
			torchvision.transforms.Resize((224, 224), interpolation=Image.BICUBIC),
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])
		from ops.dataset import PetFaceEvalDataset
		dataset = PetFaceEvalDataset(root_dir, class_name=name, transform=transform)
		datasets.append(dataset)
	dataset_len = [len(dataset) for dataset in datasets]
	dataset = torch.utils.data.ConcatDataset(datasets)
	from torch.utils.data import Dataset, DataLoader
	dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=16)
	all_features, all_labels = [], []
	import tqdm
	for images1, images2, labels in tqdm.tqdm(dataloader):
		with torch.no_grad():
			def ext(images):
				e = model(images)['features'] + model(torch.flip(images, dims=(3,)))['features']
				import torch.nn.functional as F
				e = F.normalize(e, p=2, dim=1)
				return e
			e1 = ext(images1.cuda())
			e2 = ext(images2.cuda())
			all_features.append(torch.stack([e1, e2], dim=1))
			all_labels.append(labels)
	all_features = torch.cat(all_features, dim=0).split(dataset_len, dim=0)
	all_labels = torch.cat(all_labels, dim=0).split(dataset_len, dim=0)

	all_results = {}
	for idx, name in enumerate(tqdm.tqdm(class_names)):
		features = all_features[idx]
		labels = all_labels[idx]
		if len(features) == 0:
			continue
		embeddings1, embeddings2 = features[:, 0].cpu().numpy(), features[:, 1].cpu().numpy()
		import numpy as np
		thresholds = np.arange(0, 4, 0.01)
		from ops.evaluation import calculate_roc
		results = calculate_roc(thresholds,
											embeddings1,
											embeddings2,
											np.asarray(labels),
											nrof_folds=10)
		all_results[name + '_auc'] = results['auc']
		all_results[name + '_acc'] = results['accuracy']
	all_results['mean_auc'] = np.mean([all_results[k] for k in all_results if k.endswith('_auc')])
	all_results['mean_acc'] = np.mean([all_results[k] for k in all_results if k.endswith('_acc')])
	return all_results
	# for k in all_results:
	# 	print(k, all_results[k])
		# print(catnames[name], results['accuracy'], results['auc'])


# Function to calculate Top-1 accuracy and mean average precision (mAP) from similarity lists
def calculate_top1_map(similarity_list, labels):
	top1_count = 0
	avg_precision = 0
	num_queries = len(similarity_list)
	for i in range(num_queries):
		sims = similarity_list[i]
		query_label = labels[i]
		adjusted_labels = labels[:i] + labels[i+1:]
		relevant_indices = [j for j, label in enumerate(adjusted_labels) if label == query_label]
		top1_index = torch.argmax(sims).item()
		if top1_index in relevant_indices:
			top1_count += 1
		sorted_indices = torch.argsort(sims, descending=True)
		precision_at_i = 0
		correct_at_i = 0
		num_relevant = len(relevant_indices)
		for j, index in enumerate(sorted_indices):
			if index in relevant_indices:
				correct_at_i += 1
				precision_at_i += correct_at_i / (j + 1)
		if num_relevant > 0:
			avg_precision += precision_at_i / num_relevant
	top1_accuracy = top1_count / num_queries
	map_score = avg_precision / num_queries
	return top1_accuracy, map_score


# Define a custom Dataset class
class ImageListDataset(Dataset):
	def __init__(self, image_list, transform=None):
		self.image_list = image_list
		self.transform = transform

	def __len__(self):
		return len(self.image_list)

	def __getitem__(self, idx):
		img_path = self.image_list[idx]
		image = Image.open(img_path)
		if self.transform:
			image = self.transform(image)
		return image


def evaluate_cute(model, object_classes, dataset_path):
	if object_classes is None:
		object_classes = sorted([name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))])
	image_list = []
	for object_class in tqdm.tqdm(object_classes):
		image_list.extend(sorted(glob.glob(os.path.join(dataset_path, object_class, "instance_1/**/*.jpg"))))
		image_list.extend(sorted(glob.glob(os.path.join(dataset_path, object_class, "instance_2/**/*.jpg"))))
	len(image_list), image_list[0]
	transform = torchvision.transforms.Compose([
		# torchvision.transforms.Resize((336, 336), interpolation=Image.BICUBIC),
		torchvision.transforms.Resize((224, 224), interpolation=Image.BICUBIC),
		torchvision.transforms.ToTensor(),
		# torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	dataset = ImageListDataset(image_list=image_list, transform=transform)
	dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=16)
	all_features = {}
	count = 0
	for images in tqdm.tqdm(dataloader):
		# outputs = model(images.cuda(), output_hidden_states=True)
		# outputs = projection(encoder.forward_features(images.cuda())['x_prenorm'][:, 0])
		outputs = model(images.cuda())['features']
		# outputs = projection(encoder(images.cuda()))
		for i in range(len(images)):
			r = outputs[i]
			r = r / r.norm()
			all_features[image_list[count]] = r.cpu()
			count += 1

	def eval(images1, images2):
		labels = [0] * len(images1) + [1] * len(images2)
		features = torch.stack([all_features[p] for p in images1 + images2])

		features = torch.nn.functional.normalize(features, dim=1)
		sims = 1 - torch.cdist(features, features, p=2) * 0.5
		diag_mask = torch.eye(sims.size(0), dtype=bool)
		non_diag_mask = ~diag_mask
		sims = sims[non_diag_mask].reshape(sims.size(0), sims.size(1) - 1)

		top1_accuracy, map_score = calculate_top1_map(sims, labels)
		return top1_accuracy, map_score

	all_top1_accuracies = []
	all_map_scores = []
	for object_class in tqdm.tqdm(object_classes):
		images1 = sorted(glob.glob(os.path.join(dataset_path, object_class, "instance_1/in_the_wild", "*.jpg")))
		images2 = sorted(glob.glob(os.path.join(dataset_path, object_class, "instance_2/in_the_wild", "*.jpg")))
		top1_accuracy, map_score = eval(images1, images2)
		all_top1_accuracies.append(top1_accuracy)
		all_map_scores.append(map_score)
	in_the_wild_top1_accuracy = np.mean(all_top1_accuracies)
	in_the_wild_map_score = np.mean(all_map_scores)

	all_top1_accuracies = []
	all_map_scores = []
	for object_class in tqdm.tqdm(object_classes):
		for i in range(24):
			images1 = []
			images2 = []
			for name in ['light_back', 'light_left', 'light_low', 'light_right']:
				l1 = sorted(glob.glob(os.path.join(dataset_path, object_class, "instance_1/"+name, "*.jpg")))
				l2 = sorted(glob.glob(os.path.join(dataset_path, object_class, "instance_2/"+name, "*.jpg")))
				if len(l1) != len(l2):
					# print("Different number of images in", object_class, name)
					continue
				images1.append(l1[i])
				images2.append(l2[i])
			top1_accuracy, map_score = eval(images1, images2)
			all_top1_accuracies.append(top1_accuracy)
			all_map_scores.append(map_score)
	light_top1_accuracy = np.mean(all_top1_accuracies)
	light_map_score = np.mean(all_map_scores)

	all_top1_accuracies = []
	all_map_scores = []
	for object_class in tqdm.tqdm(object_classes):
		for name in ['light_back', 'light_left', 'light_low', 'light_right']:
			images1 = sorted(glob.glob(os.path.join(dataset_path, object_class, "instance_1/"+name, "*.jpg")))
			images2 = sorted(glob.glob(os.path.join(dataset_path, object_class, "instance_2/"+name, "*.jpg")))
			top1_accuracy, map_score = eval(images1, images2)

			all_top1_accuracies.append(top1_accuracy)
			all_map_scores.append(map_score)
	pose_top1_accuracy = np.mean(all_top1_accuracies)
	pose_map_score = np.mean(all_map_scores)

	return {'in_the_wild_top1_accuracy': in_the_wild_top1_accuracy, 'in_the_wild_map_score': in_the_wild_map_score,
			'light_top1_accuracy': light_top1_accuracy, 'light_map_score': light_map_score,
			'pose_top1_accuracy': pose_top1_accuracy, 'pose_map_score': pose_map_score}


def compute_rank1_rank5_map(features, labels):
    """
    Compute Rank-1, Rank-5 accuracy, and Mean Average Precision (mAP).
    
    :param features: Tensor of shape (N, D), where N is the number of samples and D is the feature dimension.
    :param labels: Tensor of shape (N,), containing the class labels for each sample.
    :return: rank1_acc, rank5_acc, mean_AP
    """

    # Normalize features to unit length (useful for cosine similarity)
    features = torch.nn.functional.normalize(features, dim=1)

    # sorted_indices = []
    # bs = 256
    # for batch_indices, batch_features in zip(torch.arange(features.size(0)).long().split(bs, dim=0), features.split(bs, dim=0)):
    #     print()
    #     similarity_matrix = torch.matmul(batch_features, features.T)  # (N, N)
    #     similarity_matrix[torch.arange(similarity_matrix.size(0)).long(), batch_indices] = -float("inf")  # Ensures self-match is ignored
    #     # sorted_indices.append(torch.topk(similarity_matrix, dim=1, k=5).indices)
    #     sorted_indices.append(torch.argsort(similarity_matrix, dim=1, descending=True))
        
    # Compute similarity matrix using cosine similarity
    # similarity_matrix = torch.matmul(features, features.T)  # (N, N)
    # Remove self-matching by setting diagonal values to -inf
    N = labels.size(0)
    # identity_mask = torch.eye(N, dtype=torch.bool, device=features.device)
    # similarity_matrix[identity_mask] = -float("inf")  # Ensures self-match is ignored
    # Sort indices based on similarity scores (descending order, most similar first)
    # sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
    # sorted_indices = sorted_indices[:, :-1]
    # sorted_indices = torch.topk(similarity_matrix, dim=1, k=5).indices
    # sorted_indices = torch.cat(sorted_indices, dim=0)
    # sorted_indices = sorted_indices[:, :-1]
    # print(sorted_indices.size())
    # print(sorted_indices)
    # print(sorted_indices)

    # Compute Rank-1 and Rank-5 accuracy
    # rank1_correct = (labels[sorted_indices[:, 0]] == labels).float().mean()  # Top-1 match
    # rank5_correct = (labels[sorted_indices[:, :5]] == labels.unsqueeze(1)).any(dim=1).float().mean()  # Top-5 match

    rank1_correct, rank5_correct = 0, 0
    ap_list = []
    for i in tqdm.trange(N):
        similarity_matrix = (features[i].unsqueeze(0) * features).sum(1)  # (N)
        similarity_matrix[i] = -float("inf")  # Ensures self-match is ignored
        sorted_indices = torch.argsort(similarity_matrix, dim=0, descending=True)[:-1]

        rank1_correct += (labels[sorted_indices[0]] == labels[i]).item()
        rank5_correct += (labels[sorted_indices[:5]] == labels[i]).any().item()
        
        # Identify correct matches in the sorted list
        relevant = labels[sorted_indices] == labels[i]
        # print(relevant)

        if relevant.sum() == 0:  # Skip if there are no correct matches
            continue

        # print(sorted_indices[i].size())
        # Compute Precision@k for relevant positions
        # precisions = torch.cumsum(relevant, dim=0) / torch.arange(1, N, device=features.device).float()
        precisions = torch.cumsum(relevant, dim=0) / torch.arange(1, N, device=features.device).float()
        
        # Compute Average Precision (AP) for the current query
        ap = (precisions * relevant).sum() / relevant.sum()
        ap_list.append(ap)

    # Compute mean AP (mAP)
    mean_AP = torch.stack(ap_list).mean().item() if ap_list else 0.0

    return {'rank1': rank1_correct * 100 / N, 'rank5': rank5_correct * 100 / N, 'mAP': mean_AP * 100}

def evaluate_reid(model, class_names, root_dir):
	model.eval()
	from ops.dataset import PetFaceEvalDataset
	datasets = []
	for name in class_names:
		import torchvision
		from PIL import Image
		transform = torchvision.transforms.Compose([
			torchvision.transforms.Resize((224, 224), interpolation=Image.BICUBIC),
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])
		from ops.dataset import PetFaceReIDEvalDataset
		dataset = PetFaceReIDEvalDataset(root_dir, class_name=name, transform=transform)
		datasets.append(dataset)
	dataset_len = [len(dataset) for dataset in datasets]
	dataset = torch.utils.data.ConcatDataset(datasets)
	from torch.utils.data import Dataset, DataLoader
	dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=16)
	all_features, all_labels = [], []
	import tqdm
	for images1, labels in tqdm.tqdm(dataloader):
		with torch.no_grad():
			def ext(images):
				e = model(images)['features'] + model(torch.flip(images, dims=(3,)))['features']
				import torch.nn.functional as F
				e = F.normalize(e, p=2, dim=1)
				return e
			e1 = ext(images1.cuda())
			all_features.append(e1)
			all_labels.append(labels)
	all_features = torch.cat(all_features, dim=0).split(dataset_len, dim=0)
	all_labels = torch.cat(all_labels, dim=0).split(dataset_len, dim=0)

	all_results = {}
	for idx, name in enumerate(tqdm.tqdm(class_names)):
		features = all_features[idx]
		labels = all_labels[idx]
		if len(features) == 0:
			continue
		# from ops.evaluation import compute_rank1_rank5_map
		results = compute_rank1_rank5_map(features, labels.cuda())
		# print(name, results)
		for k in results:
			all_results[name + '_' + k] = results[k]
	import numpy as np
	mean_rank1 = np.mean([all_results[k] for k in all_results if k.endswith('_rank1')])
	mean_rank5 = np.mean([all_results[k] for k in all_results if k.endswith('_rank5')])
	mean_mAP = np.mean([all_results[k] for k in all_results if k.endswith('_mAP')])
	return {'mean_rank1': mean_rank1, 'mean_rank5': mean_rank5, 'mean_mAP': mean_mAP}