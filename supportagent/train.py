import datetime
import os
import pandas as pd
import re
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from datasets import ZendeskDataset
from pprint import pprint
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
from zenpy import Zenpy
from zenpy.lib.api_objects import Comment, Ticket
from zenpy.lib.exception import RecordNotFoundException
from model import Model


def train(model, epochs=3):
    df = pd.read_csv('zendesktickets.csv')
    df = df.sample(frac=1.)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    ds = ZendeskDataset(df)

    dataloader = DataLoader(ds, batch_size=1)

    for epoch in range(0, epochs):
        running_loss = 0.
        for b, batch in enumerate(dataloader):
            X, y = batch
            X = torch.tensor(X).to(device)
            y = torch.tensor(y).to(device)

            output = model(X)
            next_token = output.logits[-1, :]
            loss = criterion(next_token.reshape(1, -1), y)
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch} Batch {b} Running Loss {running_loss / (b + 1)}')
        torch.save(model.state_dict(), f'Model_{epoch}_{running_loss / (b + 1):.2f}')


if __name__ == "__main__":
    zendesk_model = Model()
    train(zendesk_model, epochs=3)
