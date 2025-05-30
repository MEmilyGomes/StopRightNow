<div align="center">
  <img src="Imagens/logo_Ilum-CNPEM.png" alt="Descrição da imagem" width="1000"/>
</div>

<h1 align="center"> Stop right now, thank you very much 🤚💃:</h1>
<h2 align="center">Implementação da  estratégia de Parada Antecipada</h2>

<p align="center"><strong>Autoras:</strong> Lorena Ribeiro Nascimento e Maria Emily Nayla Gomes da Silva</p>
<p align="center"><strong>Orientador:</strong> Prof. Dr. Daniel R. Cassar</p>


<p align="center">
<img loading="lazy" src="http://img.shields.io/static/v1?label=STATUS&message=EM%20DESENVOLVIMENTO&color=GREEN&style=for-the-badge"/>
</p>


## 📝 Descrição
<p align="justify">
Nesse projeto, implementou-se a técnica de Parada Antecipada (<i>Early Stopping</i>) no treinamento de uma rede neural convolucional (CNN) implementada em PyTorch com os dados do dataset MNIST. Essa técnica avalia o erro de validação em cada época de treinamento e, caso ocorra um aumento do erro — ou seja, esse valor seja superior ao intervalo máximo permitido, definido pelo hiperparâmetro <code>DELTA_MINIMO</code> — durante a quantidade máxima de vezes definida pela <code>PACIENCIA</code>, o treinamento será interrompido antes de completar todas as épocas previamente estabelecidas. O objetivo dessa técnica é garantir que o modelo interrompa o treinamento ao atingir um ponto de estabilidade, evitando <i>overfitting</i> e custo computacional desnecessário.
</p>

## 📔 Notebooks e arquivos do projeto
* `Imagens/logo_Ilum-CNPEM.png` — Logo da instituição Ilum - CNPEM.  
* `EarlyStopping.ipynb` — Caderno Jupyter principal do projeto.  
* `README.md` — Descrição geral do projeto.

## 🏋️‍♀️ Implementando a Parada Antecipada (Early Stopping)
<p align="justify">
Para implementar a parada antecipada, criou-se uma classe <code>EarlyStopper</code>, a qual, ao instanciar uma variável, salva as informações dos hiperparâmetros <code>DELTA_MINIMO</code> e <code>PACIENCIA</code>. A classe apresenta um método, o <code>early_stop</code> que avalia a perda de validação a cada época e registra quando essa perda aumenta em relação à anterior. Caso o erro aumente por mais épocas do que o valor definido por <code>PACIENCIA</code>, o método retorna verdadeiro e o treinamento da rede é interrompido.
</p>

`````Python
class EarlyStopper:
    def __init__(self, PACIENCIA, DELTA_MINIMO):
        self.patience = PACIENCIA
        self.min_delta = DELTA_MINIMO
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, perda_validacao):
        if perda_validacao < self.min_validation_loss:
            self.min_validation_loss = perda_validacao
            self.counter = 0
        elif perda_validacao >= (self.min_validation_loss + self.min_delta): 
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
`````

## 😁 Conclusão
<p align="justify">
A parada antecipada foi devidamente implementada, sendo definido o treinamento por 25 épocas, mas interrompido na 23ª. A CNN, implementada com PyTorch e utilizando os dados do MNIST, atingiu uma acurácia de 99%. Observou-se que a definição do <i>delta</i> mínimo é bastante relevante e deve ser ajustado conforme o comportamento da rede. No nosso caso, como a variação do erro ocorria em valores muito baixos, o <i>delta</i> foi proporcional a essa variação. Portanto, sendo uma implementação simples da técnica, que resultou em um modelo não sobreajustado e com consumo eficiente de recursos computacionais, conclui-se que a parada antecipada é uma técnica bastante relevante para ser aplicada.
</p>

## 🖇️ Informações técnicas
* Linguagem de programação: `Python 3.9`
* Software:  `Jupyter Notebook`
* **Bibliotecas e Módulos:** `torch`, `torchvision`, `torchvision.datasets`, `torchvision.transforms`, `torch.utils.data.random_split`, `torch.utils.data.DataLoader`, `torch.nn`, `torch.nn.functional`, `torch.optim`, `matplotlib.pyplot`, `os`


## 🧠 Contribuições dos Colaboradores
| [<img loading="lazy" src="https://avatars.githubusercontent.com/u/172424739?v=4" width=115><br><sub>Lorena Ribeiro Nascimento</sub>](https://github.com/Lorena881)<br> [<sub>Ilum - CNPEM</sub>](https://ilum.cnpem.br/)<br> [<sub>Currículo Lattes</sub>]()<br> [<sub>Linkedin</sub>]() | [<img loading="lazy" src="https://avatars.githubusercontent.com/u/172424897?v=4" width=115><br><sub> Maria Emily Nayla</sub>](https://github.com/MEmilyGomes)<br> [<sub>Ilum - CNPEM</sub>](https://ilum.cnpem.br/)<br> [<sub>Currículo Lattes</sub>](http://lattes.cnpq.br/9482558334105708)<br> | [<img loading="lazy" src="https://github.com/user-attachments/assets/463d4753-7fa4-4a42-aa54-409e4150bb51" width=115><br> <sub> Prof. Dr. Daniel R. Cassar </sub>](https://github.com/drcassar)<br> [<sub>Ilum - CNPEM</sub>](https://ilum.cnpem.br/)<br> [<sub>Currículo Lattes</sub>](http://lattes.cnpq.br/1717397276752482) | 
| :---: | :---: | :---: | 

#### Para o Projeto:
* Lorena Ribeiro: Construção da CNN, implementação da parada antecipada e comentários no notebook principal.
* Emily Gomes: Construção da CNN e implementação da parada antecipada.

#### Para o Repositório GitHub:
* Emily Gomes: README.

**Orientação:** Prof. Dr. Daniel R. Cassar.
