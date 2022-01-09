import jsonlines
import os 
import json 
from collections import OrderedDict, defaultdict
import numpy as np
import torch
from cli import parse_args
import logging
import argparse
from tqdm import tqdm
from utils import extract_clusters, extract_mentions_to_predicted_clusters_from_clusters, extract_clusters_for_decode
from modeling import S2E
from transformers import AutoTokenizer, LongformerConfig, AutoConfig
import tokenizations
from itertools import chain


logger = logging.getLogger(__name__)


class Inference:
    def __init__(self, args, model) -> None:
        self.args = args 
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-large-4096')
        self.model = model 
        self.model.eval()
        self.input_file = args.input_file
        logger.info("Loading input file...")
        with jsonlines.open(self.input_file, 'r') as f:
            self.data = [doc for doc in f]

        print(1)

    def long2origin(self,doc):
        long2origin = []
        long_doc_tokens = []
        tokens = doc['tokens']
        for i in range(len(tokens)):
            if i == 0 or tokens[i][0] == '’':
                token = tokens[i]
            else:
                token = ' ' + tokens[i]
            longformer_token = self.tokenizer.tokenize(token)
            long_doc_tokens.extend(longformer_token)
            long2origin += [i]*len(longformer_token)
        return long2origin, long_doc_tokens

    def doc_from_tokens(self, doc):
        document = ' '.join(doc['tokens'])
        document = document.replace(' ’', '’')
        return document


    def predict(self):

        logger.info(f"***** Running inference on {len(self.data)} documents *****")

        for i, doc in enumerate(tqdm(self.data)):

            if 'clusters' in doc.keys():
                continue
            try:
                longformer_tokens = self.tokenizer.tokenize(self.doc_from_tokens(doc))
                _, long2origin_org = tokenizations.get_alignments(doc['tokens'], longformer_tokens)
                long2origin, long_doc_tokens= self.long2origin(doc)
                assert longformer_tokens == long_doc_tokens, 'longform_tokens different from long_doc_tokens'
                assert len(long2origin) == len(longformer_tokens),  'long2origin length differnet of longformer_tokens length'
                input_ids = self.tokenizer.encode(longformer_tokens[:4094], return_tensors='pt').to(self.args.device)
                attention_mask = torch.ones(input_ids.shape).to(self.args.device)
                long2origin = [0] + long2origin
                long2origin.append(long2origin[-1])

                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    return_all_outputs=True)

                outputs_np = tuple(tensor.cpu().numpy() for tensor in outputs)
                for starts, end_offsets, coref_logits, _ in zip(*outputs_np):
                    max_antecedents = np.argmax(coref_logits, axis=1).tolist()
                    mention_to_antecedent = {((int(start), int(end)), (int(starts[max_antecedent]), int(end_offsets[max_antecedent]))) for start, end, max_antecedent in
                                             zip(starts, end_offsets, max_antecedents) if max_antecedent < len(starts)}

                    predicted_clusters, _ = extract_clusters_for_decode(mention_to_antecedent)
                    origin_clusters = [[(long2origin[start], long2origin[end]) for start, end in cluster] \
                                        for cluster in predicted_clusters]
                    self.data[i]['clusters'] = origin_clusters
            except:
                continue

        with jsonlines.open(self.input_file, 'w') as f:
            f.write_all(self.data)



def main():
    args = parse_args()
    args.device = "cuda:{}".format(args.gpu) if torch.cuda.is_available() else 'cpu'
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    
    logger.info("Loading model")
    config_class = LongformerConfig
    base_model_prefix = "longformer"
    config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    S2E.config_class = config_class
    S2E.base_model_prefix = base_model_prefix
    model = S2E.from_pretrained(args.model_name_or_path,
                                config=config,
                                cache_dir=args.cache_dir,
                                args=args)
    model.to(args.device)

    inference = Inference(args, model)
    inference.predict()


if __name__=='__main__':
    main()
