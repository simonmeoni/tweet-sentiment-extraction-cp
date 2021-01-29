def count_parameters(mdl):
    return sum(p.numel() for p in mdl.parameters() if p.requires_grad)


def jaccard_score(str1, str2):
    l_str1 = set(str1.lower().split())
    l_str2 = set(str2.lower().split())
    inter = l_str1.intersection(l_str2)
    return float(len(inter)) / (len(l_str1) + len(l_str2) - len(inter))
