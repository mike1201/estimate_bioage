<1> Imputation.py : bioage.csv --> bioage_V1.csv
<2> Imputation.R : bioage_V1.csv --> bioage_m_ImputationR.csv and bioage_f_R.csv
<3> change_numerical_to_categorical.py : bioage_m_ImputationR.csv and bioage_f_R.csv --> bioage_m_change, bioage_f_change
<4> divide_age_var.py : bioage_m_change, bioage_f_change --> bioage_change_divide

# First Clustering
<5> clustering_once.py : bioage_change_divide --> bioage_m_change_divide_groupscore, bioage_f_change_divide_groupscore

# Comparing with KNN
<6> compare_group_first.py : bioage_m_change_divide_groupscore, bioage_f_change_divide_groupscore --> bioage_change_divided_group1_compare1

# Second Clustering
<7> clustering_twice.py : bioage_change_divided_group1_compare1 --> bioage_m_change_divided_group1_compare1_group2, bioage_f_change_divided_group1_compare1_group2

# Comparing with KNN
<8> compare_group_twice.py : bioage_m_change_divided_group1_compare1_group2, bioage_f_change_divided_group1_compare1_group2 --> bioage_change_divided_group1_compare1_group2_compare2

# Get bioage
<9> get_bioage.py : bioage_change_divided_group1_compare1_group2_compare2 --> estimated_bioage, estimated_variable_bioage