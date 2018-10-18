import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision.datasets.folder import default_loader
from sklearn.model_selection import train_test_split


from nima.train.utils import SCORE_NAMES, TAG_NAMES


def _remove_all_not_found_image(df: pd.DataFrame, path_to_images: str) -> pd.DataFrame:
    clean_rows = []
    for _, row in df.iterrows():
        image_id = row['image_id']
        try:
            _ = default_loader(os.path.join(path_to_images, f"{image_id}.jpg"))
        except (FileNotFoundError, OSError):
            pass
        else:
            clean_rows.append(row)
    df_clean = pd.DataFrame(clean_rows)
    return df_clean


def remove_all_not_found_image(df: pd.DataFrame, path_to_images: str, num_workers: int = 64) -> pd.DataFrame:
    futures = []
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for df_batch in np.array_split(df, num_workers):
            future = executor.submit(_remove_all_not_found_image, df=df_batch, path_to_images=path_to_images)
            futures.append(future)
        for future in tqdm(as_completed(futures)):
            results.append(future.result())
    new_df = pd.concat(results)
    return new_df


def _read_ava_txt(path_to_ava: str) -> pd.DataFrame:
    df = pd.read_csv(path_to_ava, header=None, sep=' ')
    del df[0]
    scores_names = SCORE_NAMES
    tag_names = TAG_NAMES
    df.columns = ['image_id'] + scores_names + tag_names
    return df


def clean_and_split(path_to_ava_txt: str, path_to_save_csv: str, path_to_images: str):
    df = _read_ava_txt(path_to_ava_txt)
    df = remove_all_not_found_image(df, path_to_images)

    df_train, df_val_test = train_test_split(df, train_size=0.9)
    df_val, df_test = train_test_split(df_val_test, train_size=0.5)

    df_train.to_csv(os.path.join(path_to_save_csv, 'train.csv'), index=False)
    df_val.to_csv(os.path.join(path_to_save_csv, 'val.csv'), index=False)
    df_test.to_csv(os.path.join(path_to_save_csv, 'test.csv'), index=False)
