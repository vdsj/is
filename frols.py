# tentativa de implementação do frols

import numpy as np
import matplotlib.pyplot as plt

def frols_algorithm(y, u, max_order):
    n = len(y)
    order = 0  # Comece com um modelo vazio
    model_terms = []  # Mantenha uma lista dos índices dos termos selecionados
    X = np.ones((n, 1))  # Inicialize a matriz de design com um termo constante
    beta = np.zeros(1)  # Inicialize os coeficientes
    
    while order < max_order and order < u.shape[1]:  # Evite exceder o número de colunas em u
        best_term_index = None
        best_residual = None
        
        for i in range(len(u[0])):
            if i not in model_terms:
                term = u[:, i]
                X_temp = np.column_stack((X, term))
                
                # Calcule os coeficientes usando regressão linear
                beta_temp = np.linalg.inv(X_temp.T @ X_temp) @ X_temp.T @ y
                
                # Calcule o erro residual
                residual = y - X_temp @ beta_temp
                
                if best_residual is None or np.linalg.norm(residual) < np.linalg.norm(best_residual):
                    best_term_index = i
                    best_residual = residual
                    best_beta = beta_temp
        
        # Adicione o melhor termo ao modelo
        model_terms.append(best_term_index)
        X = np.column_stack((X, u[:, best_term_index]))
        
        # Ortogonalização dos termos
        for j in range(order):
            term_j = u[:, model_terms[j]]
            projection = np.dot(X[:, -1].T, term_j) / np.dot(term_j.T, term_j)
            X[:, -1] -= projection * term_j
        
        beta = best_beta
        order += 1
        
        # Plote os valores reais e os valores previstos pelo modelo
        plt.figure()
        plt.plot(y, label="Real")
        plt.plot(X @ beta, label="Previsto")
        plt.xlabel("Tempo")
        plt.ylabel("Valor")
        plt.title(f"Modelo NARMAX - Ordem {order}")
        plt.legend()
        plt.show()
    
    return beta, model_terms

# Exemplo de uso
#np.random.seed(0)
n = 10
t = np.arange(n)


# Dados de entrada u (primeira coluna)
dados = np.loadtxt('robot_arm.dat')

size = 60 

u = np.reshape(dados[:size, 0], (len(dados[:size, 0]), 1))
y = np.reshape(dados[:size, 1], (len(dados[:size, 1]), 1))


max_order = 1000  # Ordem máxima do modelo
beta, model_terms = frols_algorithm(y, u, max_order)
print("Coeficientes estimados:", beta)
print("Índices dos termos selecionados:", model_terms)
