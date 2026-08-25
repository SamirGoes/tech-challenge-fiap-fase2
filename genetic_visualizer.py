import json
import os
import sys
import time

import pygame


class GeneticVisualizer:

    #Cores (fundo escuro, linha verde, texto branco)
    COR_FUNDO = (18, 18, 18)
    COR_GRAFICO_FUNDO = (30, 30, 30)
    COR_LINHA = (0, 220, 0)
    COR_LINHA_PONTO = (0, 255, 0)
    COR_EIXOS = (90, 90, 90)
    COR_TEXTO = (255, 255, 255)
    COR_TEXTO_DESTAQUE = (0, 220, 0)

    def __init__(self, largura=1100, altura=780, total_geracoes=50,
                 titulo="Evolução do Algoritmo Genético", altura_grafico_max=None):
        pygame.init()
        pygame.display.set_caption(titulo)

        self.largura = largura
        self.altura = altura
        #Janela redimensionável: o usuário pode arrastar a borda para aumentar
        #o espaço caso os textos não caibam na tela.
        self.tela = pygame.display.set_mode((self.largura, self.altura), pygame.RESIZABLE)

        #Fontes
        self.fonte_normal = pygame.font.SysFont("consolas", 20)
        self.fonte_titulo = pygame.font.SysFont("consolas", 24, bold=True)

        #Dados do AG
        self.total_geracoes = total_geracoes
        self.historico_fitness = []          #melhor fitness por geração
        self.melhor_fitness_global = None
        self.melhores_params_global = {}
        self.geracao_atual = 0

        #Margens fixas em torno do gráfico. A margem inferior (onde ficam os
        #textos) é recalculada dinamicamente conforme a quantidade de
        #hiperparâmetros, então não corta informação mesmo com muitos genes.
        self.margem_esquerda = 70
        self.margem_direita = 30
        self.margem_topo = 60
        self.altura_linha_texto = 26
        self.altura_grafico_max = altura_grafico_max

        self.grafico_rect = pygame.Rect(0, 0, 0, 0)
        self._recalcular_layout()

        self.clock = pygame.time.Clock()
        self.rodando = True

        #Botão "< Voltar": permite ao usuário retornar à tela de seleção de
        #arquivo sem fechar a janela. Fica visível durante a animação e na
        #tela final de resultado.
        self.voltar_solicitado = False
        self.botao_voltar_hover = False

    def _margem_baixo_necessaria(self):
        linhas_fixas = 4  #geração, fitness da geração, fitness global, cabeçalho "hiperparâmetros"
        linhas_params = max(len(self.melhores_params_global), 3)  #reserva espaço mínimo
        total_linhas = linhas_fixas + linhas_params
        return 40 + total_linhas * self.altura_linha_texto

    def _recalcular_layout(self):
        margem_baixo = self._margem_baixo_necessaria()
        altura_grafico = self.altura - self.margem_topo - margem_baixo

        #Garante uma altura mínima para o gráfico mesmo em janelas pequenas
        #ou com muitos hiperparâmetros, evitando um retângulo invertido.
        altura_grafico = max(altura_grafico, 120)

        #Limita o gráfico a uma altura máxima, se configurado, deixando o
        #espaço sobrando como respiro acima do painel de texto.
        if self.altura_grafico_max is not None:
            altura_grafico = min(altura_grafico, self.altura_grafico_max)

        self.grafico_rect = pygame.Rect(
            self.margem_esquerda,
            self.margem_topo,
            self.largura - self.margem_esquerda - self.margem_direita,
            altura_grafico,
        )
        self.margem_baixo = margem_baixo

    def reiniciar(self, total_geracoes):
        self.total_geracoes = total_geracoes
        self.historico_fitness = []
        self.melhor_fitness_global = None
        self.melhores_params_global = {}
        self.geracao_atual = 0
        self.voltar_solicitado = False
        self._recalcular_layout()

    def _calcular_rect_botao_voltar(self):
        largura_botao, altura_botao = 110, 32
        margem = 15
        return pygame.Rect(
            self.largura - largura_botao - margem,
            margem,
            largura_botao,
            altura_botao,
        )

    def _desenhar_botao_voltar(self):
        rect = self._calcular_rect_botao_voltar()
        cor_fundo = self.COR_LINHA if self.botao_voltar_hover else self.COR_GRAFICO_FUNDO
        cor_texto = (10, 10, 10) if self.botao_voltar_hover else self.COR_TEXTO

        pygame.draw.rect(self.tela, cor_fundo, rect, border_radius=4)
        pygame.draw.rect(self.tela, self.COR_EIXOS, rect, width=1, border_radius=4)

        texto = self.fonte_normal.render("< Voltar", True, cor_texto)
        texto_rect = texto.get_rect(center=rect.center)
        self.tela.blit(texto, texto_rect)

    #Métodos públicos


    def update(self, geracao, melhor_fitness, melhores_params, fps=30):
        if not self.rodando:
            return False

        self.geracao_atual = geracao
        self.historico_fitness.append(melhor_fitness)

        #Atualiza o melhor resultado global (não decresce)
        if self.melhor_fitness_global is None or melhor_fitness > self.melhor_fitness_global:
            self.melhor_fitness_global = melhor_fitness
            self.melhores_params_global = dict(melhores_params)
            #Recalcula o layout: se o número de hiperparâmetros mudou, a
            #margem inferior precisa se ajustar para não cortar texto.
            self._recalcular_layout()

        self._processar_eventos()

        if self.rodando:
            self._desenhar()
            pygame.display.flip()
            self.clock.tick(fps)

        return self.rodando

    def save_image(self, caminho="evolucao_ag.png"):
        pygame.image.save(self.tela, caminho)
        print(f"[GeneticVisualizer] Imagem salva em: {caminho}")

    def close(self):
        pygame.quit()

    def selecionar_arquivo(self, pasta_resultados):
        candidatos = listar_arquivos_compativeis(pasta_resultados)
        if not candidatos:
            raise FileNotFoundError(
                f"Nenhum arquivo .json com histórico de gerações encontrado em: {pasta_resultados}"
            )

        indice_selecionado = 0
        indice_hover = None
        escolhido = False

        while self.rodando and not escolhido:
            rects_itens = self._calcular_rects_selecao(len(candidatos))

            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    self.rodando = False
                elif evento.type == pygame.VIDEORESIZE:
                    self.largura, self.altura = evento.w, evento.h
                    self.tela = pygame.display.set_mode((self.largura, self.altura), pygame.RESIZABLE)
                    rects_itens = self._calcular_rects_selecao(len(candidatos))
                elif evento.type == pygame.KEYDOWN:
                    if evento.key == pygame.K_ESCAPE:
                        self.rodando = False
                    elif evento.key == pygame.K_UP:
                        indice_selecionado = (indice_selecionado - 1) % len(candidatos)
                    elif evento.key == pygame.K_DOWN:
                        indice_selecionado = (indice_selecionado + 1) % len(candidatos)
                    elif evento.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        escolhido = True
                elif evento.type == pygame.MOUSEMOTION:
                    indice_hover = None
                    for i, rect in enumerate(rects_itens):
                        if rect.collidepoint(evento.pos):
                            indice_hover = i
                            break
                elif evento.type == pygame.MOUSEBUTTONDOWN and evento.button == 1:
                    for i, rect in enumerate(rects_itens):
                        if rect.collidepoint(evento.pos):
                            indice_selecionado = i
                            escolhido = True
                            break

            if not self.rodando:
                break

            self._desenhar_tela_selecao(candidatos, rects_itens, indice_selecionado, indice_hover)
            pygame.display.flip()
            self.clock.tick(30)

        if not self.rodando:
            return None

        return os.path.join(pasta_resultados, candidatos[indice_selecionado])

    def _calcular_rects_selecao(self, quantidade_itens):
        x = 60
        y_inicial = 130
        altura_item = 36
        largura_item = self.largura - 2 * x
        return [
            pygame.Rect(x, y_inicial + i * altura_item, largura_item, altura_item - 6)
            for i in range(quantidade_itens)
        ]

    def _desenhar_tela_selecao(self, candidatos, rects_itens, indice_selecionado, indice_hover):
        self.tela.fill(self.COR_FUNDO)

        titulo = self.fonte_titulo.render(
            "Selecione o arquivo de resultados do Algoritmo Genético", True, self.COR_TEXTO
        )
        self.tela.blit(titulo, (60, 40))

        subtitulo = self.fonte_normal.render(
            "↑ / ↓ para navegar   |   Enter para confirmar   |   clique com o mouse", True, self.COR_EIXOS
        )
        self.tela.blit(subtitulo, (60, 80))

        for i, (nome, rect) in enumerate(zip(candidatos, rects_itens)):
            if i == indice_selecionado:
                cor_fundo = self.COR_LINHA
                cor_texto = (10, 10, 10)
            elif i == indice_hover:
                cor_fundo = self.COR_GRAFICO_FUNDO
                cor_texto = self.COR_TEXTO_DESTAQUE
            else:
                cor_fundo = self.COR_GRAFICO_FUNDO
                cor_texto = self.COR_TEXTO

            pygame.draw.rect(self.tela, cor_fundo, rect, border_radius=4)
            texto = self.fonte_normal.render(nome, True, cor_texto)
            self.tela.blit(texto, (rect.x + 12, rect.y + 5))

    #Métodos internos de desenho

    def _processar_eventos(self):
        rect_botao_voltar = self._calcular_rect_botao_voltar()

        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                self.rodando = False
            elif evento.type == pygame.KEYDOWN and evento.key == pygame.K_ESCAPE:
                self.rodando = False
            elif evento.type == pygame.VIDEORESIZE:
                self.largura, self.altura = evento.w, evento.h
                self.tela = pygame.display.set_mode((self.largura, self.altura), pygame.RESIZABLE)
                self._recalcular_layout()
            elif evento.type == pygame.MOUSEMOTION:
                self.botao_voltar_hover = rect_botao_voltar.collidepoint(evento.pos)
            elif evento.type == pygame.MOUSEBUTTONDOWN and evento.button == 1:
                if rect_botao_voltar.collidepoint(evento.pos):
                    self.voltar_solicitado = True

    def _desenhar(self):
        self.tela.fill(self.COR_FUNDO)

        self._desenhar_titulo()
        self._desenhar_grafico()
        self._desenhar_textos_info()
        self._desenhar_botao_voltar()

    def _desenhar_titulo(self):
        texto = self.fonte_titulo.render(
            "Evolução do Fitness - Otimização de Random Forest (AG)", True, self.COR_TEXTO
        )
        self.tela.blit(texto, (self.margem_esquerda, 15))

    def _desenhar_grafico(self):
        pygame.draw.rect(self.tela, self.COR_GRAFICO_FUNDO, self.grafico_rect)
        pygame.draw.rect(self.tela, self.COR_EIXOS, self.grafico_rect, width=1)

        if len(self.historico_fitness) < 1:
            return

        fitness_min = min(self.historico_fitness)
        fitness_max = max(self.historico_fitness)

        #Evita divisão por zero quando todos os valores são iguais
        if abs(fitness_max - fitness_min) < 1e-9:
            fitness_min -= 0.05
            fitness_max += 0.05

        #Eixo X: usa o total de gerações esperado, ou o número já visto (o que for maior)
        eixo_x_max = max(self.total_geracoes - 1, len(self.historico_fitness) - 1, 1)

        pontos = []
        for i, fitness in enumerate(self.historico_fitness):
            x = self.grafico_rect.left + (i / eixo_x_max) * self.grafico_rect.width
            y_prop = (fitness - fitness_min) / (fitness_max - fitness_min)
            y = self.grafico_rect.bottom - y_prop * self.grafico_rect.height
            pontos.append((x, y))

        if len(pontos) >= 2:
            pygame.draw.lines(self.tela, self.COR_LINHA, False, pontos, width=2)

        #Destaca o último ponto (geração atual)
        pygame.draw.circle(self.tela, self.COR_LINHA_PONTO, (int(pontos[-1][0]), int(pontos[-1][1])), 5)

        self._desenhar_rotulos_eixos(fitness_min, fitness_max, eixo_x_max)

    def _desenhar_rotulos_eixos(self, fitness_min, fitness_max, eixo_x_max):
        #Eixo Y: min e max do fitness
        rotulo_max = self.fonte_normal.render(f"{fitness_max:.4f}", True, self.COR_TEXTO)
        rotulo_min = self.fonte_normal.render(f"{fitness_min:.4f}", True, self.COR_TEXTO)
        self.tela.blit(rotulo_max, (self.grafico_rect.left - 65, self.grafico_rect.top - 8))
        self.tela.blit(rotulo_min, (self.grafico_rect.left - 65, self.grafico_rect.bottom - 8))

        #Eixo X: geração 0 e geração final
        rotulo_x0 = self.fonte_normal.render("0", True, self.COR_TEXTO)
        rotulo_xn = self.fonte_normal.render(str(int(eixo_x_max)), True, self.COR_TEXTO)
        self.tela.blit(rotulo_x0, (self.grafico_rect.left - 5, self.grafico_rect.bottom + 8))
        self.tela.blit(rotulo_xn, (self.grafico_rect.right - 20, self.grafico_rect.bottom + 8))

        #Rótulos dos eixos
        label_y = self.fonte_normal.render("Fitness", True, self.COR_TEXTO)
        label_x = self.fonte_normal.render("Geração", True, self.COR_TEXTO)
        self.tela.blit(label_x, (self.grafico_rect.centerx - 30, self.grafico_rect.bottom + 30))

    def _desenhar_textos_info(self):
        y = self.grafico_rect.bottom + 55
        x = self.margem_esquerda

        linhas = [
            (f"Geração: {self.geracao_atual} / {self.total_geracoes}", self.COR_TEXTO),
            (f"Melhor fitness da geração: {self.historico_fitness[-1]:.4f}"
             if self.historico_fitness else "Melhor fitness da geração: -", self.COR_TEXTO),
            (f"Melhor fitness global: {self.melhor_fitness_global:.4f}"
             if self.melhor_fitness_global is not None else "Melhor fitness global: -",
             self.COR_TEXTO_DESTAQUE),
            ("Melhores hiperparâmetros encontrados:", self.COR_TEXTO),
        ]

        for texto, cor in linhas:
            superficie = self.fonte_normal.render(texto, True, cor)
            self.tela.blit(superficie, (x, y))
            y += 26

        #Lista os hiperparâmetros, um por linha
        for nome_param, valor in self.melhores_params_global.items():
            texto_param = f"   {nome_param}: {valor}"
            superficie = self.fonte_normal.render(texto_param, True, self.COR_TEXTO_DESTAQUE)
            self.tela.blit(superficie, (x, y))
            y += 24


#==========================================================================
#CARREGAMENTO DOS DADOS A PARTIR DE UM ARQUIVO JSON
#==========================================================================

def load_json(caminho):
    with open(caminho, "r", encoding="utf-8") as arquivo:
        dados = json.load(arquivo)

    geracoes = dados.get("geracoes", [])
    geracoes.sort(key=lambda item: item["geracao"])
    return geracoes


def listar_arquivos_compativeis(pasta_resultados):
    if not os.path.isdir(pasta_resultados):
        raise FileNotFoundError(f"Pasta de resultados não encontrada: {pasta_resultados}")

    candidatos = []
    for nome in sorted(os.listdir(pasta_resultados)):
        if not nome.endswith(".json"):
            continue
        caminho_completo = os.path.join(pasta_resultados, nome)
        try:
            with open(caminho_completo, "r", encoding="utf-8") as arquivo:
                if "geracoes" in json.load(arquivo):
                    candidatos.append(nome)
        except (json.JSONDecodeError, OSError):
            continue

    return candidatos


def escolher_arquivo_json(pasta_resultados):
    candidatos = listar_arquivos_compativeis(pasta_resultados)
    if not candidatos:
        raise FileNotFoundError(
            f"Nenhum arquivo .json com histórico de gerações encontrado em: {pasta_resultados}"
        )

    print(f"\nArquivos de resultado disponíveis em '{pasta_resultados}':")
    for i, nome in enumerate(candidatos, start=1):
        print(f"  [{i}] {nome}")

    while True:
        escolha = input(f"Escolha o arquivo (1-{len(candidatos)}): ").strip()
        if escolha.isdigit() and 1 <= int(escolha) <= len(candidatos):
            return os.path.join(pasta_resultados, candidatos[int(escolha) - 1])
        print("Opção inválida, tente novamente.")


def localizar_pasta_resultados():
    pasta_script = os.path.dirname(os.path.abspath(__file__))

    candidatos = [
        os.path.join("experiments", "results"),                                    #relativo ao cwd
        os.path.join(pasta_script, "experiments", "results"),                       #ao lado do script
        os.path.join(pasta_script, "tech-challenge-fiap-fase2", "experiments", "results"),
    ]

    for candidato in candidatos:
        if os.path.isdir(candidato):
            return candidato

    raise FileNotFoundError(
        "Não encontrei a pasta 'experiments/results' em nenhum dos locais esperados:\n"
        + "\n".join(f"  - {c}" for c in candidatos)
        + "\nPasse o caminho manualmente: python genetic_visualizer.py <caminho>"
    )


if __name__ == "__main__":
    #Pasta com os resultados reais do AG. Pode ser sobrescrita passando o
    #caminho como argumento na linha de comando:
    #python genetic_visualizer.py caminho/para/experiments/results
    PASTA_RESULTADOS = sys.argv[1] if len(sys.argv) > 1 else localizar_pasta_resultados()

    #Abre a janela (fica aberta durante toda a execução). O laço externo
    #permite voltar à tela de seleção de arquivo (botão "< Voltar") sem
    #precisar reiniciar o script.

    visualizer = GeneticVisualizer(total_geracoes=1, altura_grafico_max=300)

    while visualizer.rodando:
        CAMINHO_JSON = visualizer.selecionar_arquivo(PASTA_RESULTADOS)
        if CAMINHO_JSON is None:
            break  #janela fechada durante a seleção

        geracoes = load_json(CAMINHO_JSON)
        visualizer.reiniciar(total_geracoes=len(geracoes))

        #Reproduz a evolução na visualização, geração por geração

        for item in geracoes:
            melhor_fitness = item["melhor_fitness"]
            melhores_params = item["melhores_params"]

            #Atualiza a visualização (não bloqueia o loop)
            continuar = visualizer.update(
                geracao=item["geracao"],
                melhor_fitness=melhor_fitness,
                melhores_params=melhores_params,
            )

            if not continuar or visualizer.voltar_solicitado:
                break

            print(f"Geração {item['geracao']:02d}/{visualizer.total_geracoes} - "
                  f"Melhor fitness: {melhor_fitness:.4f} - Params: {melhores_params}")

            #Pequena pausa para tornar a evolução visível geração a geração.
            #Substitua/remova conforme a velocidade real do seu AG.
            time.sleep(0.4)

        if not visualizer.rodando:
            break
        if visualizer.voltar_solicitado:
            continue  #volta direto para a tela de seleção de arquivo

        #Salva a imagem final e aguarda o usuário: fechar a janela,
        #ou clicar em "< Voltar" para escolher outro arquivo.

        visualizer.save_image("evolucao_ag_resultado.png")
        print("\nMelhores hiperparâmetros encontrados:", visualizer.melhores_params_global)
        print(f"Melhor fitness: {visualizer.melhor_fitness_global:.4f}")

        while visualizer.rodando and not visualizer.voltar_solicitado:
            visualizer._processar_eventos()
            visualizer._desenhar()
            pygame.display.flip()
            visualizer.clock.tick(30)

    visualizer.close()
    sys.exit()
