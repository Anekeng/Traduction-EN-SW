# Traduction de l'ANGLAIS AU SWAHILI
*************************************
##  Ce répo contient : 
1- Le notebook code_traduction qui permet d'exécuter le code pas à pas 
2- Le fichier Mon_code.py qui permet d'exécuter le code sur streamlit
3- Le dossier saved_models contenant le modèle entrainé
4- Les fichiers textes corpus et corpus2 obtenu à partir de la base
5- Les fichiers json TokenizerEN (pour l'Anglais) et TokenizerSW (pour le Swahili)
6- Le fichier requirements.txt contenant les bibliothèques nécessaires

### Fonctionnement du code
La base utilisée peut être téchargée sur Hugging Face en utilisant le code ci-dessous:
    from datasets import load_dataset
    ds = load_dataset("emuchogu/swahili-english-translation")

Par la suite il faut préparer les données, entrainer le modèles et le sauvegarder en utilisant le notebook "code_traduction" 


Enfin pour effectuer la traduction sur streamlit, il faut exécuter le fichier "Mon_code.py" dans le terminal en utilisant la commande :
            streamlit run Mon_code.py