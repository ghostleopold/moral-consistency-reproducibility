using Pkg; Pkg.activate(".")
using MoralConsistency, DataFrames, CSV

df = CSV.read("universal_properties.csv", DataFrame)

names(df)

# Set C == is_conditional (already defined)

df.set_C .= df.is_conditional

# Set A == is in C and is_self_approving

df.set_A .= df.set_C .& df.is_self_approving

# Set A1 == is in C and is maxmially self is_self_approving

df.set_A1 .= df.set_C .& df.is_maximally_self_approving

# Set K == is in C and is self-cooperative

df.set_K .= df.set_C .& df.is_self_cooperative

# Set R == intersection of A and K

df.set_R .= df.set_A .& df.set_K

# Set R1 == intersection of A1 and K

df.set_R1 .= df.set_A1 .& df.set_K

# Set R2 == Member of R1 such that pGB == Defect & dBGDD == Bad & dGGDC == Bad

df.set_R2 = fill(false, nrow(df))
for row in eachrow(df)
    if row.set_R1 == false
        continue
    else
        dBGDD = MoralConsistency.getd(MoralConsistency.Strategy(row.p, row.d), 0, 1, 0, 0)
        dGGDC = MoralConsistency.getd(MoralConsistency.Strategy(row.p, row.d), 1, 1, 0, 1)
        pGB = MoralConsistency.getp(MoralConsistency.Strategy(row.p, row.d), 1, 0)
        row.set_R2 = row.set_R1 && dBGDD == 0 && dGGDC == 0 && pGB == 0
    end
end

# Set R3 == Member of R2 such that dGBCC == Bad

df.set_R3 = fill(false, nrow(df))
for row in eachrow(df)
    if row.set_R2 == false
        continue
    else
        dGBCC = MoralConsistency.getd(MoralConsistency.Strategy(row.p, row.d), 1, 0, 1, 1)
        row.set_R3 = row.set_R2 && dGBCC == 0
    end
end

# Set R4 == Member of R3 such that pBB == C & dBBDC == Bad

df.set_R4 = fill(false, nrow(df))
for row in eachrow(df)
    if row.set_R3 == false
        continue
    else
        pBB = MoralConsistency.getp(MoralConsistency.Strategy(row.p, row.d), 0, 0)
        dBBDC = MoralConsistency.getd(MoralConsistency.Strategy(row.p, row.d), 0, 0, 0, 1)
        row.set_R4 = row.set_R3 && pBB == 1 && dBBDC == 0
    end
end

# Define the columns without using broadcasting
df[!, :is_set_Mc] = (df[!, :p] .== 11) .& df[!, :is_consistent_discriminating]
df[!, :is_set_Md] = (df[!, :p] .== 10) .& df[!, :is_consistent_discriminating]
df[!, :is_set_Mu] = (df[!, :p] .== 8) .& df[!, :is_consistent_discriminating]
df[!, :is_set_Mo] = (df[!, :p] .== 9) .& df[!, :is_consistent_discriminating]

function add_leading_eight_labels!(df)
    # Add a new :label column initialized with empty strings
    df.label = fill("", nrow(df))
    
    # Define functions to check each strategy
    function is_L1_standing_int(p_int, d_int)
        return p_int == 11 && d_int == 53196
    end
    
    function is_L2_consistent_standing_int(p_int, d_int)
        return p_int == 11 && d_int == 50124
    end
    
    function is_L3_simple_standing_int(p_int, d_int)
        return p_int == 10 && d_int == 53199
    end
    
    function is_L4_SS_SJ_int(p_int, d_int)
        return p_int == 10 && d_int == 53187
    end
    
    function is_L5_SJ_SS_int(p_int, d_int)
        return p_int == 10 && d_int == 50127
    end
    
    function is_L6_stern_judging_int(p_int, d_int)
        return p_int == 10 && d_int == 50115
    end
    
    function is_L7_staying_int(p_int, d_int)
        return p_int == 10 && d_int == 53184
    end
    
    function is_L8_judging_int(p_int, d_int)
        return p_int == 10 && d_int == 50112
    end
    
    function is_all_defect(p_int, d_int)
        return p_int == 0 && d_int == 0
    end

    function is_all_cooperate(p_int, d_int)
        return p_int == 15 && d_int == 0
    end

    function is_image_score(p_int, d_int)
        return p_int == 10 && d_int == 52428
    end

    # Iterate through each row and apply the labeling
    for i in 1:nrow(df)
        p_int = df.p[i]
        d_int = df.d[i]
        
        if is_L1_standing_int(p_int, d_int)
            df.label[i] = "L1"
        elseif is_L2_consistent_standing_int(p_int, d_int)
            df.label[i] = "L2"
        elseif is_L3_simple_standing_int(p_int, d_int)
            df.label[i] = "L3"
        elseif is_L4_SS_SJ_int(p_int, d_int)
            df.label[i] = "L4"
        elseif is_L5_SJ_SS_int(p_int, d_int)
            df.label[i] = "L5"
        elseif is_L6_stern_judging_int(p_int, d_int)
            df.label[i] = "L6"
        elseif is_L7_staying_int(p_int, d_int)
            df.label[i] = "L7"
        elseif is_L8_judging_int(p_int, d_int)
            df.label[i] = "L8"
        elseif is_all_defect(p_int, d_int)
            df.label[i] = "AllD"
        elseif is_all_cooperate(p_int, d_int)
            df.label[i] = "AllC"
        elseif is_image_score(p_int, d_int)
            df.label[i] = "Image score"
        end
    end
    return df
end

add_leading_eight_labels!(df)

# Select only the columns we want to keep
# selected_columns = [:p, :d, :set_C, :set_A, :set_A1, :set_K, :set_R, :set_R1, :set_R2, :set_R3, :set_R4, :label]
# df_selected = df[:, selected_columns]

# Write the selected columns to CSV
CSV.write("venn_data.csv", df)