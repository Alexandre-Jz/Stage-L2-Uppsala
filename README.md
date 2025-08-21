# Stage-L2-Uppsala

Le fichier ConDiSim/Vilar contient les codes de ConDiSim liés à l'oscillateur génétique.

MSE.py : calcul la MSE 
compare : fichier brouillon
run_vilar.sh : je ne m'en suis pas servi
test_timeseries : fichier de test dont je ne me suis également pas servi

vilar_dataset : fichier contenant notamment le modèle

Baseline : 

vilar_diffusion_base
vilar_main_base

vilar_main_base_optuna : permet la recherches de paramètres optimaux avec Optuna


Pipeline n°1 (FiLM + time embedding + autoencodeur):

generate_dataset : générer des datasets avec l'autoencodeur
vila_autoencodeur : fichier contenant l'autoencodeur
vilar_diffusion ; modèle de diffusion avec FiLM et Time embedding
plot_posterior : permet de plot les courbes
vilar_main : fichier principal de la pipeline n°1

vilar_main_optuna : permet la recherches de paramètres optimaux avec Optuna

Pipeline n°2 (FiLM + time embedding + feature extractor):

noencod_gen_dataset : générer des datasets de données brutes sans autoencodeur
noencod_vilar_diffusion : modèle de diffusion avec FiLM + Time embedding + Feature Extractor
noencod_feature_extractor : extracteur de features qui agit sur un dataset brut
noencod_vilar_main : fichier principal du pipeline n°2
noencod_plot_posterior : permet de plot les courbes

noencod_vilar_main_optuna : permet la recherches de paramètres optimaux avec Optuna



    
