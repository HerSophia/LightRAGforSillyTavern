from hashlib import sha256

from Levenshtein import ratio


def calculate_similarity(keyword, target):
    similarity = ratio(keyword, target)

    if (keyword == "第一魂技"):
        print("keyword:", target)
        print("target_name:",target)
        print("ratio:",similarity)
    return similarity,similarity >= 0.7  # 如果相似度大于等于 0.8，返回 True

if __name__ == "__main__":
    similarity,is_related= calculate_similarity('第一魂技',"第一魂技")
    print(similarity)
    print(is_related)