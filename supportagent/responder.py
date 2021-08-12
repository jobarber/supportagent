import argparse
import sys
import torch
from utils import needs_comment, respond_to_ticket, get_new_tickets
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer, BertForMaskedLM, \
    BertTokenizer

def respond(model_path, tokenizer_path):
    model = torch.load(model_path)
    #tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    new_tickets = get_new_tickets()
    for ticket in new_tickets:
        if needs_comment(ticket):
            respond_to_ticket(ticket.id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='respond to tickets')
    parser.add_argument('--model_path', metavar='N', type=str, nargs=1,
                        help='path to trained model')
    parser.add_argument('--tokenizer_path', metavar='N', type=str, nargs=1,
                        help='path to tokenizer')

    args = parser.parse_args()
    print(args)
    respond(args.model_path[0], args.tokenizer_path[0])
