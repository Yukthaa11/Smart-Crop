def final_severity(leaf_pct, soil_fertility_class):
    if soil_fertility_class == "Low":
        soil_score = 30
    elif soil_fertility_class == "Medium":
        soil_score = 15
    else:
        soil_score = 5

    final = 0.7 * leaf_pct + 0.3 * soil_score
    return final
