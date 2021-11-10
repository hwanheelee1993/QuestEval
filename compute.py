from questeval.questeval_metric import QuestEval
import os
import pickle
import numpy as np
import torch
import argparse
import difflib
d = difflib.Differ()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--org_file", default="bottomup.txt", type=str)
    parser.add_argument("--corrected_file", default="bottomup_corrected.txt", type=str)
    parser.add_argument("--org_score_file", default="org_questeval.pkl", type=str)
    parser.add_argument("--output_file", default="output.pkl", type=str)
    parser.add_argument("--datadir", default='data/', type=str)
    parser.add_argument("--batch_size", default=16, type=int)

    args = parser.parse_args()
    
    with open(os.path.join(args.datadir, 'CNNDM_test_article.txt'), 'r') as f:
        sources = f.read().splitlines()

    with open(os.path.join(args.datadir, args.org_file), 'r') as f:
        org_summaries = f.read().splitlines()
        
    with open(os.path.join(args.datadir, args.corrected_file), 'r') as f:
        cor_summaries = f.read().splitlines()
        
    with open(os.path.join(args.datadir, args.org_score_file), 'rb') as f:
        org_scores = pickle.load(f)
    print("### Data Loaded")
    # Find indexes of the summaries that are edited through correction model.
    is_cors = []
    for org_sum, cor_sum in zip(org_summaries, cor_summaries):
        diff = d.compare(org_sum.lower().split(), cor_sum.lower().split())
        diff_list = list([x for x in diff])
        is_cor = False
        for x in diff_list:
            if (x.startswith('+') or x.startswith('-')):
                is_cor = True
        if(is_cor):
            is_cors.append(1)
        else:
            is_cors.append(0)
    # Compute the scores for the edited summaries
    source_picked = [x for i,x in enumerate(sources) if is_cors[i] == 1]
    cor_preds = [x for i,x in enumerate(cor_summaries) if is_cors[i] == 1]
    
    print("Num of Edited: ", np.sum(is_cors))        

    questeval = QuestEval(no_cuda=False)

    scores = questeval.corpus_questeval(
        hypothesis=cor_preds, 
        sources=source_picked, batch_size=args.batch_size)
    
    # Merge the scores with the summaries without editing
    edited_scores = []
    j = 0
    for i in range(len(org_scores['ex_level_scores'])):
        if is_cors[i] == 1:
            edited_scores.append(scores['ex_level_scores'][j])
            j += 1
        else:
            edited_scores.append(org_scores['ex_level_scores'][i])
    print("Original Score: ", np.average(org_scores['ex_level_scores']))
    print("Score After Correction: ", np.average(edited_scores))
    
    with open(os.path.join(args.datadir, args.output_file), 'wb') as f:
        pickle.dump(edited_scores, f)
                    
if __name__ == "__main__":
    main()        
