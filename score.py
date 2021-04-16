def get_model(model_name, key_file):
    if model_name.lower() in ['gpt2', 'gpt2-s', 'gpt2-small', 'gs', 's', 'small']:
        # GPT-2 Small
        model   = GPT2LMHeadModel.from_pretrained('gpt2').cuda(0).eval()
        encoder = GPT2Tokenizer.from_pretrained('gpt2')
        name    = 'G-S'
    elif model_name.lower() in ['gpt2-m', 'gpt2-medium', 'gm', 'm', 'medium']:
        # GPT-2 Medium
        model   = GPT2LMHeadModel.from_pretrained('gpt2-medium').cuda(0).eval()
        encoder = GPT2Tokenizer.from_pretrained('gpt2-medium')
        name    = 'G-M'
    elif model_name.lower() in ['gpt2-l', 'gpt2-large', 'gl', 'l', 'large']:
        # GPT-2 Large
        model   = GPT2LMHeadModel.from_pretrained('gpt2-large').cuda(0).eval()
        encoder = GPT2Tokenizer.from_pretrained('gpt2-large')
        name    = 'G-L'
    elif model_name.lower() in ['gpt2-xl', 'gxl', 'xl', 'extra-large']:
        # GPT-2 XL
        model   = GPT2LMHeadModel.from_pretrained('gpt2-xl').cuda(0).eval()
        encoder = GPT2Tokenizer.from_pretrained('gpt2-xl')
        name    = 'G-XL'
    elif model_name.lower() == 'ada' or \
         model_name.lower() == 'babbage' or \
         model_name.lower() == 'curie' or \
         model_name.lower() == 'davinci':
        # GPT-3
        model = name = model_name
        encoder = None
        import openai
        with open(key_file) as f:
            api_key = f.read().strip()
        openai.api_key = api_key
    else:
        raise ValueError(f'No model {model_name}')
    return model, encoder, name

def get_examples(dataset_name, split, stem):
    if dataset_name == 'copa':
        from data_loaders import load_examples_copa
        examples = load_examples_copa(f'{stem}copa-{split}.xml')
        closed_label_space = False
    elif dataset_name == 'copa-rev':
        from data_loaders import load_examples_copa_rev
        examples = load_examples_copa_rev(f'{stem}copa-{split}.xml')
        closed_label_space = False
    elif dataset_name == 'storycloze':
        from data_loaders import load_examples_storycloze
        examples = load_examples_storycloze(f'{stem}{split}.tsv')
        closed_label_space = False
    elif dataset_name == 'hellaswag':
        from data_loaders import load_examples_hellaswag
        examples = load_examples_hellaswag(f'{stem}dev.jsonl')
        closed_label_space = False
    elif dataset_name == 'race-m' or \
         dataset_name == 'race-h':
        from data_loaders import load_examples_race
        version = 'high' if dataset_name == 'race-h' else 'middle'
        examples = load_examples_race(stem, split, version)
        closed_label_space = False
    elif dataset_name == 'arc-easy' or \
         dataset_name == 'arc-challenge':
        from data_loaders import load_examples_arc
        examples = load_examples_arc(f'{stem}{split}.jsonl')
        closed_label_space = False
    elif dataset_name == 'obqa':
        from data_loaders import load_examples_obqa
        examples = load_examples_obqa(f'{stem}{split}.jsonl')
        closed_label_space = False
    elif dataset_name == 'cqa':
        from data_loaders import load_examples_cqa
        if args.split == 'test':
            raise NotImplementedError("CSQA does not release test answers directly, please do not spam their leaderboard either :)")
        else:
            examples = load_examples_cqa(f'{stem}{split}.jsonl')
        closed_label_space = False
    elif dataset_name == 'boolq':
        from data_loaders import load_examples_boolq
        examples = load_examples_boolq(f'{stem}dev.jsonl')
        closed_label_space = True
    elif dataset_name == 'rte':
        from data_loaders import load_examples_rte
        examples = load_examples_rte(f'{stem}dev.jsonl')
        closed_label_space = True
    elif dataset_name == 'cb':
        from data_loaders import load_examples_cb
        examples = load_examples_cb(f'{stem}dev.jsonl')
        closed_label_space = True
    elif dataset_name == 'sst-2':
        from data_loaders import load_examples_sst2
        examples = load_examples_sst2(f'{stem}{split}.txt')
        closed_label_space = True
    elif dataset_name == 'sst-5':
        from data_loaders import load_examples_sst5
        examples = load_examples_sst5(f'{stem}{split}.txt')
        closed_label_space = True
    elif dataset_name == 'agn':
        from data_loaders import load_examples_agn
        split = 'train' if split == 'dev' else split
        examples = load_examples_agn(f'{stem}{split}.csv')
        closed_label_space = True
    elif dataset_name == 'trec':
        split = 'train' if split == 'dev' else split
        from data_loaders import load_examples_trec
        examples = load_examples_trec(f'{stem}{split}.txt')
        closed_label_space = True
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')

    return examples, closed_label_space


if __name__ == '__main__':
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from utils import score
    import argparse
    import random
    import numpy as np
    import torch
    import os
    import pdb

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('--model', type=str, default='xl')
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--batch', type=int, default=10)
    parser.add_argument('--sample', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--key', type=str, default='api.key')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    print(args)

    if args.debug:
        pdb.set_trace()

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model, encoder, name = get_model(args.model, args.key)
    if args.dataset.endswith('-rev'):
        stem = f'data/{args.dataset[:-4]}/'
    else:
        stem = f'data/{args.dataset}/'
    examples, closed_label_space = get_examples(args.dataset, args.split, stem)
    if args.sample:
        assert(args.sample <= len(examples))
        examples = random.sample(examples, args.sample)
    accs = score(model, args.model, encoder, examples, stem, args.split, args.batch)

    # print results
    print(f'{name} gets {accs}% on {args.dataset}')
    print(f"{accs['uncond']} & {accs['lm']} & {accs['tok_mean']} & {accs['dcpmi']}")
