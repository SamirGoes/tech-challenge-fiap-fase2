import { Component, inject, OnInit, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { PercentPipe } from '@angular/common';
import { Api, FeatureSimplificado, PredictLoteResponse, PredictResponse } from './api';
import {
  EXEMPLO_BENIGNO,
  EXEMPLO_MALIGNO,
  formularioDeChaves,
  recortarExemplo,
} from './features';

@Component({
  imports: [FormsModule, PercentPipe],
  selector: 'app-root',
  styleUrl: './app.css',
  templateUrl: './app.html',
})
export class App implements OnInit {
  private readonly api = inject(Api);

  protected readonly camposSimplificado = signal<FeatureSimplificado[]>([]);
  protected readonly algoritmoSimplificado = signal('');
  protected readonly aba = signal<'simplificado' | 'lote'>('simplificado');
  protected readonly valoresSimplificado = signal<Record<string, number | null>>({});
  protected readonly resultadoSimplificado = signal<PredictResponse | null>(null);
  protected readonly carregando = signal(false);
  protected readonly erro = signal<string | null>(null);

  protected readonly arquivoLote = signal<File | null>(null);
  protected readonly relatorio = signal<PredictLoteResponse | null>(null);
  protected readonly casoAberto = signal<number | null>(0);

  ngOnInit() {
    this.carregarSimplificado();
  }

  private carregarSimplificado() {
    this.api.listarSimplificado().subscribe({
      next: (resposta) => {
        this.camposSimplificado.set(resposta.features);
        this.algoritmoSimplificado.set(resposta.algoritmo);
        this.valoresSimplificado.set(formularioDeChaves(resposta.features.map((item) => item.chave)));
      },
      error: (falha) => {
        this.erro.set(
          this.lerErro(
            falha,
            'Não deu para carregar as medidas do exame simplificado. Rode o notebook e suba a API.',
          ),
        );
      },
    });
  }

  escolherAba(aba: 'simplificado' | 'lote') {
    this.aba.set(aba);
    this.erro.set(null);
  }

  atualizarCampo(chave: string, evento: Event) {
    const bruto = (evento.target as HTMLInputElement).value;
    this.valoresSimplificado.update((atual) => ({
      ...atual,
      [chave]: bruto === '' ? null : Number(bruto),
    }));
  }

  preencherSimplificado(exemplo: 'maligno' | 'benigno') {
    const fonte = exemplo === 'maligno' ? EXEMPLO_MALIGNO : EXEMPLO_BENIGNO;
    const chaves = this.camposSimplificado().map((campo) => campo.chave);
    this.valoresSimplificado.set(recortarExemplo(fonte, chaves));
    this.resultadoSimplificado.set(null);
    this.erro.set(null);
  }

  limparSimplificado() {
    const chaves = this.camposSimplificado().map((campo) => campo.chave);
    this.valoresSimplificado.set(formularioDeChaves(chaves));
    this.resultadoSimplificado.set(null);
    this.erro.set(null);
  }

  enviarSimplificado() {
    const features: Record<string, number> = {};
    const campos = this.camposSimplificado();
    if (!campos.length) {
      this.erro.set('As medidas do exame simplificado ainda não chegaram da API.');
      return;
    }
    for (const campo of campos) {
      const valor = this.valoresSimplificado()[campo.chave];
      if (valor === null || valor === undefined || Number.isNaN(valor)) {
        this.erro.set('Preencha as medidas do exame simplificado (ou use um exemplo).');
        return;
      }
      features[campo.chave] = valor;
    }

    this.carregando.set(true);
    this.erro.set(null);
    this.resultadoSimplificado.set(null);
    this.api.preverSimplificado(features).subscribe({
      next: (resposta) => {
        this.resultadoSimplificado.set(resposta);
        this.carregando.set(false);
      },
      error: (falha) => {
        this.erro.set(
          this.lerErro(falha, 'Não deu para chamar o exame simplificado. A API está rodando?'),
        );
        this.carregando.set(false);
      },
    });
  }

  escolherArquivo(evento: Event) {
    const arquivo = (evento.target as HTMLInputElement).files?.[0] ?? null;
    this.arquivoLote.set(arquivo);
    this.relatorio.set(null);
    this.erro.set(null);
  }

  soltarArquivo(evento: DragEvent) {
    evento.preventDefault();
    const arquivo = evento.dataTransfer?.files?.[0] ?? null;
    if (arquivo && arquivo.name.toLowerCase().endsWith('.csv')) {
      this.arquivoLote.set(arquivo);
      this.relatorio.set(null);
      this.erro.set(null);
    }
  }

  enviarLote() {
    const arquivo = this.arquivoLote();
    if (!arquivo) {
      this.erro.set('Escolha um CSV (por exemplo data/data-simplificado.csv).');
      return;
    }

    this.carregando.set(true);
    this.erro.set(null);
    this.relatorio.set(null);
    this.api.preverLoteSimplificado(arquivo).subscribe({
      next: (resposta) => {
        this.relatorio.set(resposta);
        this.casoAberto.set(resposta.resultados[0]?.linha ?? null);
        this.carregando.set(false);
      },
      error: (falha) => {
        this.erro.set(this.lerErro(falha, 'Não deu para importar o lote. A API está rodando?'));
        this.carregando.set(false);
      },
    });
  }

  abrirCaso(linha: number) {
    this.casoAberto.set(this.casoAberto() === linha ? null : linha);
  }

  percentual(valor: number) {
    return Math.round(valor * 100);
  }

  private lerErro(falha: { error?: { detail?: unknown }; message?: string }, padrao: string) {
    const detalhe = falha?.error?.detail;
    if (typeof detalhe === 'string') {
      return detalhe;
    }
    return falha?.message || padrao;
  }
}
