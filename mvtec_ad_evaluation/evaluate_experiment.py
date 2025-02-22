"""Compute evaluation metrics for a single experiment."""

__author__ = "Paul Bergmann, David Sattlegger"
__copyright__ = "2021, MVTec Software GmbH"

import json
from os import listdir, makedirs, path

import argparse
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from sklearn import metrics

import generic_util as util
from pro_curve_util import compute_pro
from roc_curve_util import compute_classification_roc


def parse_user_arguments():
    """Parse user arguments for the evaluation of a method on the MVTec AD
    dataset.

    Returns:
        Parsed user arguments.
    """
    parser = argparse.ArgumentParser(description="""Parse user arguments.""")

    parser.add_argument('--anomaly_maps_dir',
                        required=True,
                        help="""Path to the directory that contains the anomaly
                                maps of the evaluated method.""")

    parser.add_argument('--dataset_base_dir',
                        required=True,
                        help="""Path to the directory that contains the dataset
                                images of the MVTec AD dataset.""")

    parser.add_argument('--output_dir',
                        help="""Path to the directory to store evaluation
                                results. If no output directory is specified,
                                the results are not written to drive.""")

    parser.add_argument('--pro_integration_limit',
                        type=float,
                        default=0.3,
                        help="""Integration limit to compute the area under
                                the PRO curve. Must lie within the interval
                                of (0.0, 1.0].""")

    parser.add_argument('--evaluated_objects',
                        nargs='+',
                        help="""List of objects to be evaluated. By default,
                                all dataset objects will be evaluated.""",
                        choices=util.OBJECT_NAMES,
                        default=util.OBJECT_NAMES)

    args = parser.parse_args()

    # Check that the PRO integration limit is within the valid range.
    assert 0.0 < args.pro_integration_limit <= 1.0

    return args


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    optimal_threshold = thresholds[np.argmax(F1_scores)]
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
    }


def parse_dataset_files(object_name, dataset_base_dir, anomaly_maps_dir):
    """Parse the filenames for one object of the MVTec AD dataset.

    Args:
        object_name: Name of the dataset object.
        dataset_base_dir: Base directory of the MVTec AD dataset.
        anomaly_maps_dir: Base directory where anomaly maps are located.
    """
    assert object_name in util.OBJECT_NAMES

    # Store a list of all ground truth filenames.
    gt_filenames = []

    # Store a list of all corresponding anomaly map filenames.
    prediction_filenames = []

    # Test images are located here.
    test_dir = path.join(dataset_base_dir, object_name, 'test')
    gt_base_dir = path.join(dataset_base_dir, object_name, 'ground_truth')
    anomaly_maps_base_dir = path.join(anomaly_maps_dir, object_name, 'test')

    # List all ground truth and corresponding anomaly images.
    for subdir in listdir(str(test_dir)):

        if not subdir.replace('_', '').isalpha():
            continue

        # Get paths to all test images in the dataset for this subdir.
        test_images = [path.splitext(file)[0]
                       for file
                       in listdir(path.join(test_dir, subdir))
                       if path.splitext(file)[1] == '.png']

        # If subdir is not 'good', derive corresponding GT names.
        if subdir != 'good':
            gt_filenames.extend(
                [path.join(gt_base_dir, subdir, file + '_mask.png')
                 for file in test_images])
        else:
            # No ground truth maps exist for anomaly-free images.
            gt_filenames.extend([None] * len(test_images))

        # Fetch corresponding anomaly maps.
        prediction_filenames.extend(
            [path.join(anomaly_maps_base_dir, subdir, file)
             for file in test_images])

    print(f"Parsed {len(gt_filenames)} ground truth image files.")

    return gt_filenames, prediction_filenames


def calculate_au_pro_au_roc(gt_filenames,
                            prediction_filenames,
                            integration_limit):
    """Compute the area under the PRO curve for a set of ground truth images
    and corresponding anomaly images.

    In addition, the function computes the area under the ROC curve for image
    level classification.

    Args:
        gt_filenames: List of filenames that contain the ground truth images
          for a single dataset object.
        prediction_filenames: List of filenames that contain the corresponding
          anomaly images for each ground truth image.
        integration_limit: Integration limit to use when computing the area
          under the PRO curve.

    Returns:
        au_pro: Area under the PRO curve computed up to the given integration
          limit.
        au_roc: Area under the ROC curve.
        pro_curve: PRO curve values for localization (fpr,pro).
        roc_curve: ROC curve values for image level classifiction (fpr,tpr).
    """
    # Read all ground truth and anomaly images.
    ground_truth = []
    predictions = []

    print("Read ground truth files and corresponding predictions...")
    for (gt_name, pred_name) in tqdm(zip(gt_filenames, prediction_filenames),
                                     total=len(gt_filenames)):
        prediction = util.read_tiff(pred_name)
        predictions.append(prediction)

        if gt_name is not None:
            ground_truth.append(np.asarray(Image.open(gt_name)))
        else:
            ground_truth.append(np.zeros(prediction.shape))

    # Compute the PRO curve.
    pro_curve = compute_pro(
        anomaly_maps=predictions,
        ground_truth_maps=ground_truth)

    # Compute the area under the PRO curve.
    au_pro = util.trapezoid(
        pro_curve[0], pro_curve[1], x_max=integration_limit)
    au_pro /= integration_limit
    print(f"AU-PRO (FPR limit: {integration_limit}): {au_pro}")

    scores = np.asarray(predictions).flatten()
    masks = np.asarray(ground_truth).flatten()

    # Convert ground truth masks to binary labels (0 or 1)
    masks = (masks > 0).astype(int)

    img_min_scores = scores.min(axis=-1)
    img_max_scores = scores.max(axis=-1)
    scores = (scores - img_min_scores) / (img_max_scores - img_min_scores)
    # Ensure scores are in the range [0, 1]
    # If scores are not already in this range, you may need to normalize them
    # Example normalization if scores are in [0, 255]

    # Add debugging information
    print("Debugging Information:")
    print(f"Scores shape: {scores.shape}, min: {np.min(scores)}, max: {np.max(scores)}")
    print(f"Masks shape: {masks.shape}, unique values: {np.unique(masks)}")

    try:
        # Compute the pixel-level ROC curve.
        pauc_roc = roc_auc_score(masks, scores)
        print(f"Pixel-level classification AU-ROC: {pauc_roc}")
    except ValueError as e:
        print(f"Error calculating pixel-level AU-ROC: {e}")
        print("Checking input shapes and values:")
        print(f"Scores shape: {scores.shape}, min: {np.min(scores)}, max: {np.max(scores)}")
        print(f"Masks shape: {masks.shape}, unique values: {np.unique(masks)}")
        raise
    # Derive binary labels for each input image:
    # (0 = anomaly free, 1 = anomalous).
    binary_labels = [int(np.any(x > 0)) for x in ground_truth]
    del ground_truth

    # Compute the classification ROC curve.
    roc_curve = compute_classification_roc(
        anomaly_maps=predictions,
        scoring_function=np.max,
        ground_truth_labels=binary_labels)

    # Compute the area under the classification ROC curve.
    au_roc = util.trapezoid(roc_curve[0], roc_curve[1])
    print(f"Image-level classification AU-ROC: {au_roc}")

    # Return the evaluation metrics.
    return au_pro, au_roc, pro_curve, roc_curve, pauc_roc


def main():
    """Calculate the performance metrics for a single experiment on the
    MVTec AD dataset.
    """
    # Parse user arguments.
    args = parse_user_arguments()

    # Store evaluation results in this dictionary.
    evaluation_dict = dict()

    # Keep track of the mean performance measures.
    au_pros = []
    au_rocs = []
    au_Pros = []
    # Evaluate each dataset object separately.
    for obj in args.evaluated_objects:
        print(f"=== Evaluate {obj} ===")
        evaluation_dict[obj] = dict()

        # Parse the filenames of all ground truth and corresponding anomaly
        # images for this object.
        gt_filenames, prediction_filenames = \
            parse_dataset_files(
                object_name=obj,
                dataset_base_dir=args.dataset_base_dir,
                anomaly_maps_dir=args.anomaly_maps_dir)

        # Calculate the PRO and ROC curves.
        au_pro, au_roc, pro_curve, roc_curve, P_au_roc = \
            calculate_au_pro_au_roc(
                gt_filenames,
                prediction_filenames,
                args.pro_integration_limit)

        evaluation_dict[obj]['au_pro'] = au_pro
        evaluation_dict[obj]['classification_au_roc'] = au_roc
        evaluation_dict[obj]['P_au_roc'] = P_au_roc
        evaluation_dict[obj]['classification_roc_curve_fpr'] = roc_curve[0]
        evaluation_dict[obj]['classification_roc_curve_tpr'] = roc_curve[1]

        # Keep track of the mean performance measures.
        au_pros.append(au_pro)
        au_rocs.append(au_roc)
        au_Pros.append(P_au_roc)
        print('\n')

    # Compute the mean of the performance measures.
    evaluation_dict['mean_au_pro'] = np.mean(au_pros).item()
    evaluation_dict['mean_classification_au_roc'] = np.mean(au_rocs).item()
    evaluation_dict['mean_P_au_roc'] = np.mean(au_Pros).item()
    # If required, write evaluation metrics to drive.
    if args.output_dir is not None:
        makedirs(args.output_dir, exist_ok=True)

        with open(path.join(args.output_dir, 'metrics.json'), 'w') as file:
            json.dump(evaluation_dict, file, indent=4)

        print(f"Wrote metrics to {path.join(args.output_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
