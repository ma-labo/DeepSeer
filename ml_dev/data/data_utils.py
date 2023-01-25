"""
Name   : data_utils.py
Author : Zhijie Wang
Time   : 2021/7/7
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(self, ori_text, tar_text, ori_tokenizer, tar_tokenizer, max_length=50):
        self.ori_text = ori_text
        self.tar_text = tar_text
        self.max_length = max_length
        self.ori_tokenizer = ori_tokenizer
        self.tar_tokenizer = tar_tokenizer

    def __len__(self):
        return len(self.ori_text)

    def __getitem__(self, ix):
        ori_sentence = self.ori_text[ix]
        tar_sentence = self.tar_text[ix]
        ori_ids = self.ori_tokenizer(ori_sentence.strip('\n').split())
        tar_ids = self.tar_tokenizer(tar_sentence.strip('\n').split())
        len_ori = len(ori_ids)
        len_tar = len(tar_ids)
        padded_ori_ids = ori_ids + [0] * (self.max_length - len_ori)
        padded_tar_ids = tar_ids + [0] * (self.max_length - len_tar)
        attention_mask_ori = [1] * len_ori + [0] * (self.max_length - len_ori)
        attention_mask_tar = [1] * len_tar + [0] * (self.max_length - len_tar)
        padded_ori_ids_tensor, padded_tar_ids_tensor, attention_mask_ori_tensor, attention_mask_tar_tensor = map(
            torch.LongTensor, [padded_ori_ids, padded_tar_ids, attention_mask_ori, attention_mask_tar]
        )
        encoded_dict = {'ori_id': padded_ori_ids_tensor,
                        'tar_id': padded_tar_ids_tensor,
                        'ori_mask': attention_mask_ori_tensor,
                        'tar_mask': attention_mask_tar_tensor,
                        }
        return encoded_dict


class CommentDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256, data_col="comment_text", weight_col=None, identity_cols=None,
                 target="target", is_testing=False, special_token=False):
        self.df = df
        self.data_col = data_col
        self.weight_col = weight_col
        self.identity_cols = identity_cols
        self.target = target
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_testing = is_testing
        self.special_token = special_token

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, ix):
        comment_text = self.df.iloc[ix][self.data_col]
        comment_class = self.df.iloc[ix][self.target]

        weight_loss = None
        if self.weight_col is not None:
            weight_loss = self.df.iloc[ix][self.weight_col]

        identities = None
        if self.identity_cols is not None:
            identities = self.df.iloc[ix][self.identity_cols].to_numpy().astype(int)

        encoded_text = self.tokenizer.encode_plus(comment_text, add_special_tokens=self.special_token,
                                                  return_token_type_ids=True,
                                                  return_attention_mask=True, padding='max_length',
                                                  max_length=self.max_length, truncation=True)

        input_ids, attn_mask, token_type_ids = map(
            torch.LongTensor,
            [encoded_text['input_ids'],
             encoded_text['attention_mask'],
             encoded_text['token_type_ids']]
        )

        encoded_dict = {'input_ids': input_ids,
                        'attn_mask': attn_mask,
                        'token_type_ids': token_type_ids,
                        'y': comment_class,
                        }

        if self.weight_col is not None:
            encoded_dict['loss_w'] = weight_loss

        if self.identity_cols is not None:
            encoded_dict['identities'] = torch.from_numpy(identities)

        if not self.is_testing:
            target = encoded_dict['y']

        return encoded_dict
