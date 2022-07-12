# README Compétition Kore 2022

Voici le dossier de mon code pour la compétition Kaggle [Kore 2022](https://www.kaggle.com/competitions/kore-2022/overview)

je vais décrire chaque fichier et dossier pour expliquer leurs fonctions propres.

### Projet_8_Kore.ipynb
Ce notebook est la base du projet, c'est par lui que j'exécute le modèle, que j'effectue les phases d'apprentissage

### config.py
c'est dans ce fichier que sont centralisés tous les hyperparamètres que j'ai définis afin de les affiner au mieux pour améliorer le modèle

### environement.py
c'est le fichier le plus conséquent, c'est dedans qu'est la classe *KoreGymEnv* qui fait le pont entre le réseau de neurones et l'interface du jeu.

#### KoreGymEnv
je vais mieux détailler les éléments clés de cette classe.

##### step()
step est la méthode qui est appelée à chaque tour de prédiction, elle reçoit en entrée la précédente action donnée par le réseau, et elle envoie en retour l'état du plateau pour la prochaine prédiction

##### gym_to_kore_action()
cette méthode convertit le tableau de nombres que fournit le réseau de neurones en action réelle dans le jeu.
c'est un des 3 points clés d'un modèle par RL

##### obs_as_gym_state
cette propriété convertie en tableau de nombres l'état du plateau pour le tour actuel, avec la base spatiale actuelle.
c'est un des 3 points clés d'un modèle par RL

##### compute_reward()
cette méthode calcule la récompense que reçoit le modèle pour sa prédiction.
il fait :
`score_plateau_actuel - score_plateau_précédent + bonus_malus_de_fin`

#### clip_normalize()
cette fonction très pratique permet de normaliser des variables avec un intervalle d'entrée, et un intervalle de sortie

<div style="page-break-after: always;"></div>

### reward_utils.py
c'est dans ce fichier qu'est la fonction très importante `get_board_value()`
cette fonction prend en entrée un plateau de jeu, et donne en retour le score associé.
cette fonction est très importante car il faut trouver la juste façon de mesurer ce score.
c'est un des 3 points clés d'un modèle par RL

### main.py
lors de la soumission du modèle c'est ce fichier qui va être appelé pour être comparé à un adversaire.

### main_agent.zip
c'est le fichier compressé du réseau de neurones pour la soumission du modèle

### opponent.py
c'est une copie de main.py qui est utilisé pour entrainer le modèle contre lui-même.

### opponent_agent.zip
tout comme main_agent.zip c'est le réseau de neurones pour l'opponent, quand on entraine le modèle contre une copie de lui-même

### test.py
c'est un agent teste pour déjà entrainer le modèle contre un premier adversaire, mais aussi utile pour essayer des choses rapidement.

### lib
dans ce dossier sont stockées les librairies tierces qui sont nécessaires.

### opponent
dans ce dossier sont stockés tous les fichiers nécessaires à l'opposant principal de référence.