
rank_list = [(0, 2), (2, 4)]

def search_index(embs, top5_indx, top3_indx, top1_indx, idx):
    for idx_values, x in enumerate(embs):
        if idx in x['id'] and idx_values < rank_list[1][1] + 1:
            if idx_values == rank_list[0][0]:
                top1_indx += 1
                return  top5_indx, top3_indx, top1_indx
            elif idx_values > rank_list[0][0] and idx_values <=rank_list[0][1]:
                top3_indx += 1
                return top5_indx, top3_indx, top1_indx
            elif idx_values > rank_list[1][0] and idx_values <= rank_list[1][1]:
                top5_indx += 1
                return top5_indx, top3_indx, top1_indx

            return top5_indx, top3_indx, top1_indx
    return  top5_indx, top3_indx, top1_indx

