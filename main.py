
"""
********************************** Importação de bibliotecas ** **************************************
"""
import numpy as np
import timeit

"""
********************************** Lista de Variáveis Globais ***************************************
"""
"""
Modelagem do cromossomo 
"""
NUMERO_VARIAVEIS   = 1      # Número de Variáveis
LIMITE_INFERIOR    = [0]    # Limite inferior das Variáveis
LIMITE_SUPERIOR    = [15]   # Limite superior das Variáveis
TAMANHO_CROMOSSOMO = 12      # Número de bits do Cromossomo
FENOTIPO           = [12]    # Número de bits de cada gene
"""
#Parametrização do AG
"""
TAMANHO_POPULACAO  = 30     # Tamanho da População - Número de indivíduos
POSICAO_CORTE      = 7     # Posição do ponto de corte onde será realizado o cruzamento
PERCENTUAL_MUTACAO = 10     # Percentual de mutação onde define o número de alelos mutados
NUM_INDIV_SELECION = 10      # Número de indivíduos selecionados em cada geração
NUM_INDIV_ELITE    = 2      # Número de indivíduos da elite (permancerão intactos na próxima geração)
TIPO_FUNC_OBJETIVO = 'Min'  # Tipo de função objetivo (FO): 'min' ou 'max'
NUMERO_GERACOES    = 100     # Número máximo de Gerações
NGSM_SEM_MELHORIAS = 10      # Número de gerações sem melhoria (para ser usado como critério de parada)
CRITERIO_PARADA    = ['fo',0] # Critério de parada do AG: ['fo',valor] ou ['num_ger',NUM_GER] ou ['ngsm',NGSM]
"""
********************************** Critérios de Parada ***************************************
#['fo', valor]        : encerra a evolução do AG com base no valor de FO que determina a solução ótima   
#['num_ger', NUM_GER] : encerra a evolução do AG com base no número de gerações
#['ngsm', NGSM]       : encerra a evolução do AG com base no número de gerações sem melhoria
"""
fitness=[]

"""    
*****************************************************************************************************
Bloco do Algoritmo Genético (AG) - Classe ag
*****************************************************************************************************
"""
class ag:

  """    
  ************************************* Funções Base do AG ******************************************
  """
  """1° GERAÇÃO DA POPULAÇÃO"""
    # Método de Geração de População Aleatória
  def gera_pop_inicial(tam_populacao, tam_cromossomo):
      #Função randomica sendo 0 minimo e 1 máximo, com array de linhas e colunas
    pop_inicial = np.random.randint(0,2,[tam_populacao,tam_cromossomo])
    return pop_inicial


  """2° SELEÇÃO DA POPULAÇÃO PELA MELHOR PERFORMANCE"""
    # Seleção de melhores indivíduos 
  def selecao(populacao, num_ind_selecionados, fitness, tipo_fo):
        
    # seleciona os [num_selec] melhores indivíduos da popução [pop] com base na aptidão [fitness].
    # Verificação se a Função Objetivo é conhecida
    if tipo_fo[0:3].upper()!= 'MAX' and tipo_fo[0:3].upper()!= 'MIN' :
      print(tipo_fo, " não é um tipo de Função Objetivo conhecido! Use \'Min\' ou \'Max\'")
      return np.array([])
     
    # Verificação se a quantidade de individuos é máior do que o tamanho da população definida
    if num_ind_selecionados > TAMANHO_POPULACAO:
      print("O número de indivíduos selecionados não pode ser maior que o tamanho da população!")
      return np.array([])

    #inicializa a matriz para receber os pais selecionados, onde esta matriz inicializa toda zerada
    pais_selecionados = np.zeros((num_ind_selecionados,TAMANHO_CROMOSSOMO),'int')
    # Percorre os individuos
    for individuos_pai in range(num_ind_selecionados):
        # Verificação se a função objetivo é de maximização
        if tipo_fo[0:3].upper()=="MAX":
            individuo_fitness_max = np.where(fitness==np.max(fitness))
            fitness[individuo_fitness_max] = -np.inf
        else:
            #Coleta a todos os individuos com menores/melhores fitness possiveis
            individuo_fitness_max = np.where(fitness == np.min(fitness))
            fitness[individuo_fitness_max] = +np.inf
         
        individuo_fitness_max = individuo_fitness_max[0][0]
        pais_selecionados[individuos_pai, :] = populacao[individuo_fitness_max, :]
    return pais_selecionados

  """3° CROSS-OVER CRUZAMENTO DOS PAIS SELECIONADOS"""

  def cruzamento(pais_selecionados_cruzamento, posicao_ponto_corte):
    #incializa matriz para receber os novos filhos gerados a partir do cruzamento dos pais selecionados
    populacao_cruzada = np.zeros((TAMANHO_POPULACAO, TAMANHO_CROMOSSOMO), 'int')
    if posicao_ponto_corte > TAMANHO_CROMOSSOMO-1:
        print("Atenção! O ponto de corte é maior que o comprimento do cromossomo!")
        #se isso ocorrer, reposiciona o ponto de corte no meio do cromossomo
        posicao_ponto_corte = TAMANHO_CROMOSSOMO//2
    
    #inicia o "povoamento" da população (pop_cruz) a partir do cruzamento dos indivíduos selecionados
    i=0
    #se quiser recompor apenas parte da população pelo cruzamento, trocar TAM_POP por PERC_CRUZ/100*TAM_POP
    numero_individuo_cruzamento = TAMANHO_POPULACAO
    numero_individuo_selecionado = NUM_INDIV_SELECION
    for j in range(numero_individuo_selecionado):
      for k in range(numero_individuo_selecionado):
        if j != k:
          #print(i,"0:",posicao_ponto_corte,",",posicao_ponto_corte,":", pais_selecionados_cruzamento.shape[1])
          populacao_cruzada[i, 0:posicao_ponto_corte] = pais_selecionados_cruzamento[j, 0:posicao_ponto_corte]
          populacao_cruzada[i, posicao_ponto_corte:TAMANHO_CROMOSSOMO] = pais_selecionados_cruzamento[k,posicao_ponto_corte:TAMANHO_CROMOSSOMO]
          #print(populacao_cruzada[i,:])
          i+=1
          if i>= numero_individuo_cruzamento:
            break;
      if i>= numero_individuo_cruzamento:
        break;
      if i < TAMANHO_POPULACAO-1:
          for l in range(i, TAMANHO_POPULACAO):
              for c in range(TAMANHO_CROMOSSOMO):
                  populacao_cruzada[l][c] = np.random.randint(0, 2)
      return populacao_cruzada

  """4° MULTAÇÃO DO CRUZAMENTO DOS PAIS SELECIONADOS"""
  def mutacao(pop, perc_mut):
      pop_mut = pop
      tx_mut = perc_mut / 100
      if perc_mut > 100:
          print(
              "Atenção! O percentual de mutação não pode ser maior que 100% e será ajustado para 50%")
          tx_mut = 0.5
      num_alelos_mut = tx_mut * TAMANHO_POPULACAO * TAMANHO_CROMOSSOMO
      #print("num_alelos_mut = ",num_alelos_mut)
      for i in range(int(num_alelos_mut)):
          #print(ini, int(pop.shape[0]))
          l = np.random.randint(0, TAMANHO_POPULACAO)
          c = np.random.randint(0, TAMANHO_CROMOSSOMO)
          if pop_mut[l][c] == 0:
              pop_mut[l][c] = 1
          else:
              pop_mut[l][c] = 0
      return pop_mut
  """
  ***************************************************************************************************
  Funções de conversões
  ***************************************************************************************************
  """
  # esquema de decodificação mostrado no slide 15 - conversão de base 2 para base 10
  def conv_bin2dec(b):
      dec = 0
      pot = 0
      for i in range(len(b)-1, -1, -1):
          dec += b[i] * 2**pot
          pot = pot+1
      return dec

  # esquema de decodificação mostrado no slide 16 - binário para Número Inteiro (Z) no intervalo [inf,sup]
  def conv_bin2int(b, l_inf, l_sup):
      v = ag.conv_bin2dec(b)
      k = b.shape[0]
      v_int = l_inf + ((l_sup - l_inf) / (2**k - 1)) * v
      return int(v_int)

  # esquema de decodificação mostrado no slide 16 - binário para Número Real (R) no intervalo [inf,sup]
  def conv_bin2real(b, l_inf, l_sup):
      v = ag.conv_bin2dec(b)
      k = b.shape[0]
      v_real = l_inf + ((l_sup - l_inf) / (2**k - 1)) * float(v)
      return v_real

  # Calcula o tamanho do cromossomo (k) conforme mostrado nos slides 17 e 18
  def calc_tam_gene(l_inf, l_sup, p):
      calc = (l_sup - l_inf) * 10**p
      k = np.log2(calc)
      k = np.math.floor(k)+1
      return k

  """
  ***************************************************************************************************
  Ciclo de evolução do AG, conforme fluxograma
  ***************************************************************************************************
  """

  def evolucao(funcao_aptidao):
    tempo_inicio = timeit.default_timer()
    contagem_geracoes_sem_melhoria = 0
    criterio_parada = CRITERIO_PARADA
    criterio_parada_atingido = False
    tipo_fo = TIPO_FUNC_OBJETIVO.upper()
    print("\n INICIALIZANDO OTIMIZAÇÃO DO TIPO ", tipo_fo)
    individuo_melhor_fitness = 0
    if tipo_fo =='MIN':
      melhor_fitness = np.inf
    else:
      melhor_fitness = -np.inf

    contagem_geracoes = 0

    """DEBUGANDO POR PASSOS"""
    #Acessa a classe AG e insere valores no metodo de geração da 1° População
    populacao = ag.gera_pop_inicial(TAMANHO_POPULACAO, TAMANHO_CROMOSSOMO)
    print("\n 1° PASSO -  A 1° Geração Genética criada é : \n\n", populacao, "\n")

    print("\n ENTRANDO NO CICLO DE EVOLUÇÃO DO AG ")
    # Enquanto o Crtério de Parada do AG não é atingido
    print("\n Geração: ", end="")

    while(criterio_parada_atingido == False):
      #Incrementa 1 no contador de Gerações
      contagem_geracoes+=1
      print( contagem_geracoes, "\b, ", end="")
      print("\n")

      #Criação de um vetor para armazenar o fitness de cada individuo
      fitness = np.array(np.zeros(TAMANHO_POPULACAO))
      #print(fitness)
      #Calcula a aptidão de cada individuo com base na função de fitness
      for index_cromossomo in range(TAMANHO_POPULACAO):
        #O fitness recebe o valor de FO da população criada
        fitness[index_cromossomo] = funcao_aptidao(populacao[index_cromossomo])
        #Individuo e Aptidão
        print("Individuo: ", populacao[index_cromossomo],". aptidão/fitness: ", fitness[index_cromossomo])
      
      #Verificação se o tipo de problema é de minimização ou maximização para avaliação do fitness(aptidão)
      if tipo_fo =="MIN":
        #se houve melhoria, então o contador de gerações sem melhoria deve ser zerado
        #print(np.min(fitness))
        #print(np.max(fitness))
        #print(melhor_fitness)
        #print(individuo_melhor_fitness)
        #print(melhor_individuo)
        if np.min(fitness) < melhor_fitness:
          individuo_melhor_fitness = np.where(fitness == np.min(fitness))[0][0]
          melhor_fitness = fitness[individuo_melhor_fitness]
          melhor_individuo = populacao[individuo_melhor_fitness, :]
          contagem_geracoes_sem_melhoria = 0
        else:
          contagem_geracoes_sem_melhoria += 1
      else: # Se não, então é problema de maximização
        #se houve melhoria, então o contador de gerações sem melhoria deve ser zerado
        if np.max(fitness) > melhor_fitness:
          individuo_melhor_fitness = np.where(fitness == np.max(fitness))[0][0]
          melhor_fitness = fitness[individuo_melhor_fitness]
          melhor_individuo = populacao[individuo_melhor_fitness, :]
          contagem_geracoes_sem_melhoria = 0
        else:
          contagem_geracoes_sem_melhoria += 1

       #Verifica critério de parada
      if criterio_parada[0].upper() == 'FO':
          for ind in range(len(fitness)):
            
             if fitness[ind] == criterio_parada[1] or contagem_geracoes_sem_melhoria > NGSM_SEM_MELHORIAS :
                #print(fitness[ind])
                #print(criterio_parada[1])
                criterio_parada_atingido = True
                break
      elif criterio_parada[0].upper() == 'NGSM':
           if criterio_parada[1] == contagem_geracoes_sem_melhoria:
             criterio_parada_atingido = True
      else:
           if criterio_parada[1] == contagem_geracoes:
             criterio_parada_atingido = True

      # Se atingiu o citério de parada encerra o AG e retorna a melhor solução encontrada
      if criterio_parada_atingido == True:
        print("\b\b.\n")
        tempo_fim = timeit.default_timer()
        print("Tempo de Processamento = %.3f segundos \n" % float(tempo_fim-tempo_inicio))
        return[contagem_geracoes,melhor_individuo,melhor_fitness]

      # Seleção dos melhores indivíduos da população
      pais_selecionados = ag.selecao(populacao,NUM_INDIV_SELECION,fitness,TIPO_FUNC_OBJETIVO)
      
      populacao_cruzada = ag.cruzamento(pais_selecionados, POSICAO_CORTE)

      populacao_mutante = ag.mutacao(populacao_cruzada, PERCENTUAL_MUTACAO)
      populacao = populacao_mutante

    #Encerrado o ciclo de evolução do AG, apresenta-se o resultado
    return [contagem_geracoes, melhor_individuo, melhor_fitness]

"""
***************************************************************************************************
Modelagem do problema
***************************************************************************************************
"""
# Método para calculo da função aptidão
def calculo_de_aptidao(cromossomo):
  #Calculo da aptidão baseado na função objetivo
  gene_1 = cromossomo
  #Convertendo o gene para decimal
  x = ag.conv_bin2dec(gene_1)
  #Calculo da função objetivo
  fo = np.abs(2*x+10-18)
  return fo

#
res = ag.evolucao(calculo_de_aptidao)


#exibe a solução encontrada
print('Após', res[0], 'gerações, o AG encontrou a seguinte solução:')
print('Cromossomo: ', res[1])
print('Fitness: ', res[2])
x = ag.conv_bin2dec(res[1])
print('x = ', x)
print('2 * ', x, ' + 10 = 18')






