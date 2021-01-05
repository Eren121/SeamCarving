// Les tableaux numpy sont en 2D mais en 1D dans le code C
// Transforme une paire (ligne, colonne) en l'indice 1D
#define I(ligne, colonne) ((colonne) + (ligne) * (nb_colonnes))

#ifndef BARRIER
#    define BARRIER() barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); // Synchronisation.
#endif

#define K_MAX 3

__kernel void extraire_coutures(
    __global const float *energie,
    __global float *e_cumulee,
    __global int *chemins,
    __global int *coutures,
    const int nb_lignes)
{
    const int id = get_global_id(0);
    const int nb_colonnes = get_global_size(0);
    const int colonne = id;
    
    int i, j; // Variables temporaires pour respectivement une ligne et une colonne
    int e[K_MAX]; // Pour stocker l'énergie des voisins
    int kMin, kTmp;
    

    /////////////////// Calcul des énergies cumulées + chemins

    /// On initialise la dernière ligne en premier en copiant les données depuis l'image d'énergie source
    
    i = nb_lignes - 1;
    j = colonne;
    
    e_cumulee[I(i, j)] = energie[I(i, j)];
    
    BARRIER();
    // La synchronisation sera nécessaire à chaque ligne car chaque ligne à besoin de la précédente (celle du dessous)
    
    for(i = nb_lignes - 2; i >= 0; --i) {
        
        /// Cherche le voisin du desous qui a l'énergie cumulée minimale
        // (en faisant attention aux bords)

        e[0] = colonne == 0 ? INFINITY :               e_cumulee[I(i + 1, colonne - 1)];
        e[1] =                                         e_cumulee[I(i + 1, colonne)];
        e[2] = colonne == nb_colonnes - 1 ? INFINITY : e_cumulee[I(i + 1, colonne + 1)];

        kMin = 0;
        for(kTmp = 0; kTmp < K_MAX; ++kTmp) {
            if(e[kTmp] < e[kMin]) {
                kMin = kTmp;
            }
        }
        
        /// Actualisation de l'énergie cumulée, des chemins et synchronisation
        
        chemins[I(i, colonne)] = colonne + (kMin - 1); // kMin est un offset en fonction de la colonne courante, donc l'ajouter
        e_cumulee[I(i, colonne)] = e[kMin] + energie[I(i, colonne)];
        
        BARRIER();
    }

    /////////////////// Extractions des chemins des coutures depuis les chemins locaux calculés précédemment

    // à partir d'ici, on n'a plus besoin de synchroniser car les chemins locaux ne sont plus modifiés
    // et chaque thread n'a pas besoin de lire les chemins

    /**** Algorithme en Python

    m, n = image_energie.shape[:2]
    seams_locaux = extract_seams_energy_local(image_energie)
    
    seams = np.zeros(shape=(n, m), dtype=np.int)
    
    j = 0

    for j in range(n): # parallèle
        seams[j, 0] = j

    for j in range(n): # parallèle
        for i in range(1, m):
            seams[j, i] = seams_locaux[i - 1, seams[j, i - 1]]
    */

    i = 0;
    j = colonne;
    
    coutures[I(j, i)] = j; // Le premer pixel du chemin d'une couture est toujours celui de la colonne

    for(i = 1; i < nb_lignes; ++i) { // Reconstruire le parcours itérativement
        coutures[I(j, i)] = chemins[I(i - 1, coutures[I(j, i - 1)])];
    }
}