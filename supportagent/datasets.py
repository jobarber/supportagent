from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer

class ZendeskDataset(IterableDataset):
    def __init__(self, df):
        super(ZendeskDataset, self).__init__()
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')

    def __iter__(self):
        for i, row in self.df.iterrows():
            user_comment = row['comment'] + ' <|endoftext|> '
            agent_comment = row['comment_next']
            user_tokens = self.tokenizer.encode(user_comment, add_special_tokens=False)
            agent_tokens = self.tokenizer.encode(agent_comment, add_special_tokens=False)
            for i in range(len(agent_tokens)):
                if i + 1 < len(agent_tokens):
                    combined = user_tokens + agent_tokens[:i]
                    decoded = self.tokenizer.decode(combined)
                    X = self.tokenizer.encode(decoded)
                    y = agent_tokens[i]
                    if len(X) > 500 or i > 20:
                        continue
                    yield X, y
