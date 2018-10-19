from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision.datasets.folder import default_loader
from sklearn.model_selection import train_test_split

# SCORE_FIRST_COLUMN = 2
# SCORE_LAST_COLUMN = 12
# TAG_FIRST_COLUMN = 1
# TAG_LAST_COLUMN = 4
# SCORE_NAMES = [f'score{i}' for i in range(SCORE_FIRST_COLUMN, SCORE_LAST_COLUMN)]
# TAG_NAMES = [f'tag{i}' for i in range(TAG_FIRST_COLUMN, TAG_LAST_COLUMN)]


def _remove_all_not_found_image(df: pd.DataFrame, path_to_images: Path) -> pd.DataFrame:
    clean_rows = []
    for _, row in df.iterrows():
        image_id = row['image_id']
        try:
            _ = default_loader(path_to_images / f"{image_id}.jpg")
        except (FileNotFoundError, OSError):
            pass
        else:
            clean_rows.append(row)
    df_clean = pd.DataFrame(clean_rows)
    return df_clean


def remove_all_not_found_image(df: pd.DataFrame, path_to_images: Path, num_workers: int) -> pd.DataFrame:
    futures = []
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for df_batch in np.array_split(df, num_workers):
            future = executor.submit(_remove_all_not_found_image, df=df_batch, path_to_images=path_to_images)
            futures.append(future)
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())
    new_df = pd.concat(results)
    return new_df


def read_ava_txt(path_to_ava: Path) -> pd.DataFrame:
    # NIMA origin file format
    df = pd.read_csv(path_to_ava, header=None, sep=' ')

    del df[0]
    score_first_column = 2
    score_last_column = 12
    tag_first_column = 1
    tag_last_column = 4
    score_names = [f'score{i}' for i in range(score_first_column, score_last_column)]
    tag_names = [f'tag{i}' for i in range(tag_first_column, tag_last_column)]
    df.columns = ['image_id'] + score_names + tag_names
    # leave only score columns
    df = df[['image_id'] + score_names]
    return df


def clean_and_split(path_to_ava_txt: Path, path_to_save_csv: Path, path_to_images: Path, train_size: float,
                    num_workers: int):
    df = read_ava_txt(path_to_ava_txt)
    df = remove_all_not_found_image(df, path_to_images, num_workers=num_workers)

    df_train, df_val_test = train_test_split(df, train_size=train_size)
    df_val, df_test = train_test_split(df_val_test, train_size=0.5)

    df_train.to_csv(path_to_save_csv / 'train.csv', index=False)
    df_val.to_csv(path_to_save_csv / 'val.csv', index=False)
    df_test.to_csv(path_to_save_csv / 'test.csv', index=False)
