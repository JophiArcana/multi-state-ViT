from typing import Dict, List, Tuple

DATASETS: Dict[str, List[Tuple[str, int]]] = {
    "Common": [
        ("ILSVRC/imagenet-1k", 1000),
        ("mrm8488/ImageNet1K-val", 1000),
        ("UCSC-VLAA/Recap-COCO-30K", None),
        ("nateraw/pascal-voc-2012", None),
        ("johnowhitaker/imagenette2-320", 10),
        ("Multimodal-Fatima/CUB_train", 200),
        ("saragag/FlBirds", 7),
        ("microsoft/cats_vs_dogs", None),
        ("Robotkid2696/food_classification", 20),
    ],
    "Ego": [
        ("EgoThink/EgoThink", None),
    ],
    "Face": [
        ("nielsr/CelebA-faces", None),
        ("huggan/anime-faces", None),
    ],
    "Pose": [
        ("sayakpaul/poses-controlnet-dataset", None),
        ("razdab/sign_pose_M", None),
        ("Marqo/deepfashion-multimodal", None),
        ("Fiacre/small-animal-poses-controlnet-dataset", None),
        ("junjuice0/vtuber-tachi-e", None),
    ],
    "Hand": [
        ("trashsock/hands-images", 8),
        ("dduka/guitar-chords-v3", None),
    ],
    "Satellite": [
        ("arakesh/deepglobe-2448x2448", None),
        ("tanganke/eurosat", 10),
        ("wangyi111/EuroSAT-SAR", None),
        ("efoley/sar_tile_512", None),
    ],
    "Medical": [
        ("Mahadih534/Chest_CT-Scan_images-Dataset", None),
        ("TrainingDataPro/chest-x-rays", None),
        ("hongrui/mimic_chest_xray_v_1", None),
        ("sartajbhuvaji/Brain-Tumor-Classification", 4),
        ("Falah/Alzheimer_MRI", 4),
        ("Leonardo6/path-vqa", None),
        ("Itsunori/path-vqa_jap", None),
        ("ruby-jrl/isic-2024-2", None),
        ("VRJBro/lung_cancer_dataset", 5),
        ("keremberke/blood-cell-object-detection", None)
    ],
    "Miscs": [
        ("yashvoladoddi37/kanjienglish", None),
        ("Borismile/Anime-dataset", None),
        ("jainr3/diffusiondb-pixelart", None),
        ("jlbaker361/dcgan-eval-creative_gan_256_256", None),
        ("Francesco/csgo-videogame", None),
        ("Francesco/apex-videogame", None),
        ("huggan/pokemon", None),
        ("huggan/few-shot-universe", None),
        ("huggan/flowers-102-categories", None),
        ("huggan/inat_butterflies_top10k", None),
    ]
}




