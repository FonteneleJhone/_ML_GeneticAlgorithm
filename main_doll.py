"""
Implementação do Algoritmo Genético (AG) conforme teoria apresentada nas aulas 
Material de referência: IA_Algoritmos_Genéticos_2020-1-Alterado_04-05-2020.pptx
Autor: Sidnei Alves de Araújo 
Data: 08/05/2020
Disciplina: Inteligência Artificial
"""
"""
********************************** Importação de bibliotecas ****************************************
"""
import numpy as np
import timeit
import pandas as pd


"""
********************************** Lista de Variáveis Globais ***************************************
"""
"""
Modelagem do cromossomo 
"""
NUM_VAR = 19         #número de variáveis
L_INF = []           #limite(s) inferior(e) das variáveis
L_SUP = []           #limite(s) inferior(e) das variáveis
TAM_CROMO = 60        #Número de bits do cromossomo 
FENOTIPO = []        #Número de bits de cada gene  
"""
#Parametrização do AG
"""
TAM_POP = 20          #Tamanho da população - Número de indivíduos (soluções)
POS_PONTO_CORTE = 20  #Posição do ponto de corte (será usado no cruzamento)
PERC_MUT = 60          #Percentual de mutação (define o número de alelos mutados)
NUM_INDIV_SELEC = 5   #Número de indivíduos selecionados em cada geração
NUM_INDIV_ELITE = 14   #Número de indivíduos da elite (permancerão intactos na próxima geração)
TIPO_FO = 'Max'         #Tipo de função objetivo (FO): 'min' ou 'max'
NUM_GER = 10         #Número de gerações 
NGSM = 0             #Número de gerações sem melhoria (para ser usado como critério de parada) 
CRIT_PARADA=['fo', 100]   #Critério de parada do AG: ['fo',valor] ou ['num_ger',NUM_GER] ou ['ngsm',NGSM]
"""
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
  #Gera a população inicial de forma aleatória
  def gera_pop_incial(tam_pop, tam_crom):
      pop_ini=np.random.randint(0,2,[tam_pop,tam_crom])
      return pop_ini

  #Seleção dos melhores indivíduos
  def selecao(pop, num_indiv_selec, fitness, tipo_fo):
      # seleciona os [num_selec] melhores indivíduos da popução [pop] com base na aptidão [fitness].
      if tipo_fo[0:3].upper() != 'MAX' and tipo_fo[0:3].upper() != 'MIN' :
        print(tipo_fo, " não é um tipo de Função Objetivo conhecido! Use \'Min\' ou \'Max\'")
        return np.array([])
      if num_indiv_selec > TAM_POP:
        print("O número de indivíduos selecionados não pode ser maior que o tamanho da população!")
        return np.array([])
      #inicializa a matriz para receber os pais selecionados  
      pais_selec = np.zeros((num_indiv_selec, TAM_CROMO),'int')
      for ind_pai in range(num_indiv_selec):
          if tipo_fo[0:3].upper() == 'MAX':
              ind_fitness_max = np.where(fitness == np.max(fitness))
              fitness[ind_fitness_max] = -np.inf
          else:
              ind_fitness_max = np.where(fitness == np.min(fitness))
              fitness[ind_fitness_max] = +np.inf
          ind_fitness_max = ind_fitness_max[0][0]
          pais_selec[ind_pai, :] = pop[ind_fitness_max, :]
      return pais_selec

  #Cruzamento dos pais selecionados para recompor a população, como mostrado no slide 10 
  def cruzamento(pais_selec, pos_ponto_corte):
      #incializa matriz para receber os novos filhos gerados a partir do cruzamento dos pais selecionados
      pop_cruz = np.zeros((TAM_POP, TAM_CROMO),'int')
      if pos_ponto_corte > TAM_CROMO-1:
          print("Atenção! O ponto de corte é maior que o comprimento do cromossomo!")
          #se isso ocorrer, reposiciona o ponto de corte no meio do cromossomo 
          pos_ponto_corte = TAM_CROMO//2
      #inicia o "povoamento" da população (pop_cruz) a partir do cruzamento dos indivíduos selecionados  
      i=0
      #se quiser recompor apenas parte da população pelo cruzamento, trocar TAM_POP por PERC_CRUZ/100*TAM_POP   
      num_indiv_cruz = TAM_POP                 
      num_indiv_selec = NUM_INDIV_SELEC 
      for j in range(num_indiv_selec):
          for k in range(num_indiv_selec):
              if j != k:
                  #print(i,"0:",pos_ponto_corte,",",pos_ponto_corte,":", pais_selec.shape[1])
                  pop_cruz[i, 0:pos_ponto_corte] = pais_selec[j, 0:pos_ponto_corte]
                  pop_cruz[i, pos_ponto_corte:TAM_CROMO] = pais_selec[k, pos_ponto_corte:TAM_CROMO]
                  #print(pop_cruz[i,:])
                  i+=1
                  if i>=num_indiv_cruz:
                      break;
          if i>=num_indiv_cruz:
              break;
      #se após o cruzamento não foi possível gerar TAM_POP indivíduos, 
      #completa a população pop_cruz com indiv gerados aleatoriamente
      if i < TAM_POP-1:
          for l in range(i,TAM_POP):
              for c in range(TAM_CROMO):
                  pop_cruz[l][c]=np.random.randint(0,2)
      return pop_cruz

  #Mutação de alelos aleatórios dos indivíduos da população (ver slide 10)
  def mutacao(pop, perc_mut):
      pop_mut = pop
      tx_mut = perc_mut / 100
      if perc_mut > 100:
          print("Atenção! O percentual de mutação não pode ser maior que 100% e será ajustado para 50%")
          tx_mut = 0.5
      num_alelos_mut = tx_mut * TAM_POP * TAM_CROMO
      #print("num_alelos_mut = ",num_alelos_mut)
      for i in range(int(num_alelos_mut)):
          #print(ini, int(pop.shape[0]))
          l = np.random.randint(0,TAM_POP)
          c = np.random.randint(0,TAM_CROMO)
          if pop_mut[l][c] == 0:
              pop_mut[l][c] = 1
          else:
              pop_mut[l][c] = 0
      return pop_mut

  #Insere na proxima geração parte dos indivíduos selecionados na geração corrente
  #Essa operação é opcional no AG, sendo controlada pela variável NUM_INDIV_ELITE
  def elitismo(pais_selc, pop, num_indiv_elite):
      #verifica se o número de indivíduos da elite for maior que o número de indiv selecionados  
      num_indiv_elite = NUM_INDIV_ELITE
      if(num_indiv_elite > NUM_INDIV_SELEC):
          num_indiv_elite = NUM_INDIV_SELEC // 2
      num_indiv_pop = TAM_POP - num_indiv_elite
      #agrupa parte da dos indivídos selecionados com os n primeiros indivídos da população corrente
      #num_indiv_elite + num_indiv_pop = TAM_POP
      pop_com_elite = np.append(pais_selc[0:num_indiv_elite, :] , pop[0:num_indiv_pop, :], axis=0)
      #retorna a população com os indivíduos da elite inseridos no início
      return pop_com_elite
      
  """
  ***************************************************************************************************
  Funções diversas
  ***************************************************************************************************
  """
  # esquema de decodificação mostrado no slide 15 - conversão de base 2 para base 10 
  def conv_bin2dec(b):
      dec=0
      pot=0
      for i in range(len(b)-1,-1,-1):
          dec+=b[i] * 2**pot
          pot=pot+1
      return dec

  # esquema de decodificação mostrado no slide 16 - binário para Número Inteiro (Z) no intervalo [inf,sup]
  def conv_bin2int(b, l_inf, l_sup):
      v=ag.conv_bin2dec(b)
      k = b.shape[0]
      v_int = l_inf + ( (l_sup - l_inf) / (2**k -1) ) * v  
      return int(v_int)

  # esquema de decodificação mostrado no slide 16 - binário para Número Real (R) no intervalo [inf,sup]
  def conv_bin2real(b, l_inf, l_sup):
      v=ag.conv_bin2dec(b)
      k = b.shape[0]
      v_real = l_inf + ( (l_sup - l_inf) / (2**k -1) ) * float(v)  
      return v_real

  # Calcula o tamanho do cromossomo (k) conforme mostrado nos slides 17 e 18 
  def calc_tam_gene(l_inf, l_sup, p):
      calc = (l_sup - l_inf) * 10**p
      k = np.log2(calc) 
      k = np.math.floor(k)+1
      return k
  """
  ***************************************************************************************************
  Ciclo de evolução do AG, conforme fluxograma do slide 9
  ***************************************************************************************************
  """
      
  def evolucao(func_aptidao):
      tp_ini = timeit.default_timer()
      cont_ger_sem_melhoria = 0
      crit_parada = CRIT_PARADA
      crit_parada_atingido = False            
      tipo_fo=TIPO_FO.upper()
      print("Resolvendo um problema de otimzação do tipo: ", tipo_fo)
      ind_melhor_fitness=0
      if tipo_fo == 'MIN':
          melhor_fitness = np.inf
      else:
          melhor_fitness = -np.inf
          
      #Gera a população inicial (de forma aleatória)
      pop = ag.gera_pop_incial(TAM_POP, TAM_CROMO)

      cont_ger=0
      print("Entrando no ciclo de evolução do AG")
      #enquando o critério de parada não é atingido 
      print("Geração: ", end="")
     
      while(crit_parada_atingido == False ):    
          #incrementa 1 no contador de gerações
          cont_ger+=1
          print(cont_ger,"\b, ", end="")
          print("\n")
          
          #cria um vetor para armazenar o fitness de cada indivíduo 
          fitness = np.array(np.zeros(TAM_POP))
          
          
          
          #Calcula a aptidão de cada indivíduo com base na função de fitness
          for i in range(TAM_POP):
              fitness[i] = func_aptidao(pop[i])
             # print(" Aptidão/fitness: ", fitness[i])
              
          
          #verifica o tipo de problema para avaliar o fitness (aptidão)
          #se é um problema de minimização
          if tipo_fo == 'MIN':
              #se houve melhoria, então o contador de gerações sem melhoria deve ser zerado
              if np.min(fitness) < melhor_fitness:
                  ind_melhor_fitness = np.where(fitness == np.min(fitness))[0][0]
                  melhor_fitness = fitness[ind_melhor_fitness]
                  melhor_indiv = pop[ind_melhor_fitness,:]
                  cont_ger_sem_melhoria = 0
              else:
                  cont_ger_sem_melhoria += 1
          else: #se não é problema de maximização
              #se houve melhoria, então o contador de gerações sem melhoria deve ser zerado
              if np.max(fitness) > melhor_fitness:
                  ind_melhor_fitness = np.where(fitness == np.max(fitness))[0][0]
                  melhor_fitness = fitness[ind_melhor_fitness]
                  melhor_indiv = pop[ind_melhor_fitness,:]
                  cont_ger_sem_melhoria = 0
                  
              else:
                  cont_ger_sem_melhoria += 1
          
          #Verifica critério de parada
          if crit_parada[0].upper() == 'FO':
              for ind in range(len(fitness)):
                  if fitness[ind] == crit_parada[1] or cont_ger_sem_melhoria > NGSM: 
                    crit_parada_atingido = True 
                    break
          elif crit_parada[0].upper() == 'NGSM':
              if crit_parada[1] == cont_ger_sem_melhoria: 
                  crit_parada_atingido = True 
          else:
              if crit_parada[1] == cont_ger:
                  crit_parada_atingido = True 
                  
          #se atingiu o critério de parada encerra o AG e retorna a melhor solução encontrada        
          if crit_parada_atingido == True: 
              print("\b\b.\n")
              tp_fim = timeit.default_timer()
              print("Tempo de processamento = %.3f segundos\n" % float(tp_fim-tp_ini))
              return [cont_ger, melhor_indiv, melhor_fitness]
          
          #Seleção dos melhores indivíduos da população pop
          pais_selec = ag.selecao(pop, NUM_INDIV_SELEC, fitness, TIPO_FO)

          #Aplica o opreador de cruzamento para recompor a população
          pop_cruz = ag.cruzamento(pais_selec, POS_PONTO_CORTE)

          #Aplica o operador de mutação para gerar diversidade populacional   
          pop_mut = ag.mutacao(pop_cruz, PERC_MUT)
         
          #aplica elitismo (indivíduos a serem evados intactos para a proxima geração)
          if NUM_INDIV_ELITE > 0:
              pop_eli = ag.elitismo(pais_selec, pop_mut, NUM_INDIV_ELITE)   
              pop = pop_eli
          else:
              pop = pop_mut            
     
      #Encerrado o ciclo de evolução do AG, apresenta-se o resultado
      return [cont_ger, melhor_indiv, melhor_fitness]

"""    
*****************************************************************************************************
Fim da Classe ag - Daqui para baixo vem a modelagem do problema a ser resolvido: modelagem do cromos-
somo, parametrização do AG e definição da FO que será usada pelo AG para avaliação das soluções  
*****************************************************************************************************
"""

############base recebe os dados do arquivo diag_medico.csv (dataframe)################
uri = './DOLX21.csv'
base = pd.read_csv(uri, encoding='ansi', header=0)
cabecalho = ['hora', 'min','seg','preço','volume','agressor','classificador']

VA6=["'Leilão'","'Comprador'","'Vendedor'"]

"""
#Função para montar a regra a partir da decodificação do cromossomo do AG
"""
def  monta_regra(c):

   regra = '( '
   if c[0]==1: 
       regra += '(base["hora"][i] '
       if c[1]==0 and c[2] == 0: regra += '== '
       elif c[1]==0 and c[2] == 1: regra += '>= '
       elif c[1]==1 and c[2] == 0: regra += '<= '
       else: regra += '!= '
       ind = ag.conv_bin2dec(c[3:7])
       regra += str(ind)
       regra += ') '

   if c[8]==1:
       if len(regra) > 2: regra += 'and '
       regra += '(base["min"][i] '
       if c[9]==0 and c[10] == 0: regra += '== '
       elif c[9]==0 and c[10] == 1: regra += '>= '
       elif c[9]==1 and c[10] == 0: regra += '<= '
       else: regra += '!= '
       ind = ag.conv_bin2dec(c[11:16])
       regra += str(ind)
       regra += ') '

   if c[17]==1: 
       if len(regra) > 2: regra += 'and '
       regra += '(base["seg"][i] '
       if c[18]==0 and c[19] == 0: regra += '== '
       elif c[18]==0 and c[19] == 1: regra += '>= '
       elif c[18]==1 and c[19] == 0: regra += '<= '
       else: regra += '!= '
       ind = ag.conv_bin2dec(c[20:25])
       regra += str(ind)
       regra += ') '

   if c[26]==1: 
       if len(regra) > 2: regra += 'and '
       regra += '(base["preço"][i] '
       if c[27]==0 and c[28] == 0: regra += '== '
       elif c[27]==0 and c[28] == 1: regra += '>= '
       elif c[27]==1 and c[28] == 0: regra += '<= '
       else: regra += '!= '
       #faz o computo do valor a ser comparado com preço
       ind = ag.conv_bin2real(c[29:38], 5493, 5549.5)
       regra += str(ind)
       regra += ') '
   #verifica se o atributo volume entra na regra e o insere comparando o seu valor com VA5
   if c[39]==1: 
       if len(regra) > 2: regra += 'and '
       regra += '(base["volume"][i] '
       if c[40]==0 and c[41] == 0: regra += '== '
       elif c[40]==0 and c[41] == 1: regra += '>= '
       elif c[40]==1 and c[41] == 0: regra += '<= '
       else: regra += '!= '
       #faz o computo do valor a ser comparado com volume
       ind = ag.conv_bin2dec(c[42:54])
       regra += str(ind)
       regra += ') '
   #verifica se o atributo agressor entra na regra e o insere comparando o seu valor com VA6
   if c[55]==1:  
        if len(regra) > 2: 
            regra += 'and '
            regra += '(base["agressor"][i] '
            if c[56]==0: 
                regra += '== '
            else: 
                regra += '!= '
            ind = ag.conv_bin2dec(c[57:58])
            regra += VA6[ind]
            regra += ') '

   #finaliza a regra com )
   regra += ' ) '
   #print("A regra é :", regra) 
   #coloca o valor do último alelo na variável classe_pred
   classe_pred =  c[59]
   return regra, classe_pred 

#Função para classificar os registros da base com usando regra fornecida pelo AG

def aplica_regra_classificacao_registros_base(regra, classe_pred):
    num_acertos = 0
    num_erros = 0
    tx_acertos = 0.0

    for i in range (base.shape[0]):
        if eval(regra) and (base["classificador"][i] == classe_pred):
            num_acertos+=1
            print("Numero de Acretos",num_acertos)

        if eval(regra) and (base["classificador"][i] != classe_pred):
            num_erros+=1 

    num_reg_classe_pred = base[base['classificador']==classe_pred].shape[0]
    tx_acertos = (num_acertos-num_erros) / num_reg_classe_pred * 100
    return tx_acertos
"""
#Função para o cálculo de aptidão (fitness), com base na Função Objetivo (FO)
"""

def calc_aptidao(cromossomo):
    # obtÃ©m a regra a partir do cromossomo.
    regra, classe_pred = monta_regra(cromossomo)
    #faz a decodificaÃ§Ã£o
    #print( "REGRA :", regra, "Classe de Predição :", classe_pred)
    fo = aplica_regra_classificacao_registros_base(regra, classe_pred ) 
    return fo

#chama o AG com base na parametrizaÃ§Ã£o
res = ag.evolucao(calc_aptidao)

#exibe a solução encontrada
print('Após', res[0],'gerações, o AG encontrou a seguinte regra:')
regra, classe_pred = monta_regra(res[1])
#regra_descoberta = 'If ' + regra + 'Then' + '\n   Classe_pred = ' + classe_pred + '\n' 
#print(regra_descoberta)
print('Cromossomo: ', res[1])
print('Fitness: ', res[2])


