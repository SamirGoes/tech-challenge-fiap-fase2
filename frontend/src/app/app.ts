import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { PercentPipe } from '@angular/common';
import { Api, PredictLoteResponse, PredictResponse } from './api';
import {
  EXEMPLO_BENIGNO,
  EXEMPLO_MALIGNO,
  GRUPOS,
  formularioVazio,
} from './features';

@Component({
  imports: [FormsModule, PercentPipe],
  selector: 'app-root',
  styleUrl: './app.css',
  templateUrl: './app.html',
})
export class App {
  private readonly api = inject(Api);

  protected readonly grupos = GRUPOS;
  protected readonly aba = signal<'individual' | 'lote'>('individual');
  protected readonly valores = signal(formularioVazio());
  protected readonly resultado = signal<PredictResponse | null>(null);
  protected readonly carregando = signal(false);
  protected readonly erro = signal<string | null>(null);

  protected readonly arquivoLote = signal<File | null>(null);
  protected readonly relatorio = signal<PredictLoteResponse | null>(null);
  protected readonly casoAberto = signal<number | null>(0);

  escolherAba(aba: 'individual' | 'lote') {
    this.aba.set(aba);
    this.erro.set(null);
  }

  atualizarCampo(chave: string, evento: Event) {
    const bruto = (evento.target as HTMLInputElement).value;
    this.valores.update((atual) => ({
      ...atual,
      [chave]: bruto === '' ? null : Number(bruto),
    }));
  }

  preencher(exemplo: 'maligno' | 'benigno') {
    this.valores.set({ ...(exemplo === 'maligno' ? EXEMPLO_MALIGNO : EXEMPLO_BENIGNO) });
    this.resultado.set(null);
    this.erro.set(null);
  }

  limpar() {
    this.valores.set(formularioVazio());
    this.resultado.set(null);
    this.erro.set(null);
  }

  enviarIndividual() {
    const features: Record<string, number> = {};
    for (const [chave, valor] of Object.entries(this.valores())) {
      if (valor === null || Number.isNaN(valor)) {
        this.erro.set('Preencha todas as 30 medidas do exame (ou use um exemplo).');
        return;
      }
      features[chave] = valor;
    }

    this.carregando.set(true);
    this.erro.set(null);
    this.resultado.set(null);
    this.api.prever(features).subscribe({
      next: (resposta) => {
        this.resultado.set(resposta);
        this.carregando.set(false);
      },
      error: (falha) => {
        this.erro.set(this.lerErro(falha, 'Não deu para chamar o /predict. A API está rodando?'));
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
      this.erro.set('Escolha um CSV (por exemplo data/data-api.csv).');
      return;
    }

    this.carregando.set(true);
    this.erro.set(null);
    this.relatorio.set(null);
    this.api.preverLote(arquivo).subscribe({
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
