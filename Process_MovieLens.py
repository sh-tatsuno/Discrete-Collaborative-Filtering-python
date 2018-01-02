import numpy as np
import time
from tqdm import trange
import pandas as pd
import codecs

def load_data():
    # 評価値を持つファイルはu.data。楽に行列化するために、まずはDataFrameでロードする
    ratings = pd.read_csv('ml-100k/u.data', delimiter='\t', header=None).iloc[:, :3]
    ratings.rename(columns={0: 'user_id', 1: 'item_id', 2: 'rating'}, inplace=True)
    print('ratings.shape: {}'.format(ratings.shape)) # ratings.shape: (100000, 3)

    # item_idを行、user_idを列とする1682*943の行列(欠測値は0埋め)
    rating_matrix = ratings.pivot(index='item_id', columns='user_id', values='rating').fillna(0).as_matrix()
    print('rating_matrix.shape: {}'.format(rating_matrix.shape)) # rating_matrix.shape: (1682, 943)

    # pd.read_csvで読み込もうとすると文字コードエラーになるため、
    # codecs.openでエラーを無視してファイルを読み込んだ後にDataFrameにする
    with codecs.open('ml-100k/u.item', 'r', 'utf-8', errors='ignore') as f:
        item_df = pd.read_table(f, delimiter='|', header=None).loc[:, 0:1]
        item_df.rename(columns={0: 'item_id', 1: 'item_title'}, inplace=True)
        item_df.set_index('item_id', inplace=True)

    items = pd.Series(item_df['item_title'])
    print('items.shape: {}'.format(items.shape)) # items.shape: (1682, 1)
    return rating_matrix, items

def predict_ranking(user_index, reducted_matrix, original_matrix, n):
    # 対象ユーザの評価値
    reducted_vector = reducted_matrix[:, user_index]
    
    # 評価済みのアイテムの値は0にする
    filter_vector = original_matrix[:, user_index] == 0
    predicted_vector = reducted_vector * filter_vector

    # 上位n個のアイテムのインデックスを返す
    return [(i, predicted_vector[i]) for i in np.argsort(predicted_vector)[::-1][:n]]


def print_ranking(user_ids, items, reducted_matrix, original_matrix, n=10):
    for user_id in user_ids:
        predicted_ranking = predict_ranking(user_id - 1, reducted_matrix, original_matrix, n)
        print('User: {}:'.format(user_id))
        for item_id, rating in predicted_ranking:
            # アイテムID, 映画タイトル, 予測した評価値を表示
            print('{}: {} [{}]'.format(item_id, items[item_id + 1], rating))