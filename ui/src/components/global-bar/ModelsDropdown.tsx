// ModelsDropdown.tsx — Model selection UI for the global param bar
//
// Uses custom ModelSelect dropdown to show GGUF/SafeTensors format badges.

import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Download } from 'lucide-react';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { modelApi, trtBundleApi } from '../../services/api';
import { formatDitModel, formatLmModel, formatVaeModel, formatEmbeddingModel, getDitModelDescription, getLmModelDescription, getVaeModelDescription } from './modelLabels';
import { ModelManagerModal } from '../model-manager/ModelManagerModal';
import { ModelSelect } from './ModelSelect';
import type { AceModels } from '../../types';

interface ModelsDropdownProps {
  /** Whether the Model Manager modal is currently open. Controlled by parent
   *  so the parent can keep the BarSection locked (prevent hover-close) while
   *  the modal is visible. */
  showModelManager: boolean;
  onModelManagerOpen: () => void;
  onModelManagerClose: () => void;
}

export const ModelsDropdown: React.FC<ModelsDropdownProps> = ({
  showModelManager,
  onModelManagerOpen,
  onModelManagerClose,
}) => {
  const gp = useGlobalParams();
  const { t } = useTranslation();
  const [models, setModels] = useState<AceModels | null>(null);
  const [trtBundles, setTrtBundles] = useState<string[]>([]);

  useEffect(() => {
    modelApi.list()
      .then(setModels)
      .catch(() => {});
  }, []);

  // Available TRT bundles join the DiT options — selecting one sets ditModel,
  // which the server maps to synth_model = <bundle name> at generate time.
  const loadTrtBundles = () => {
    trtBundleApi.list()
      .then((r) => {
        const names = r.bundles.flatMap((b) => {
          const out: string[] = [];
          if (b.engines['q8map-fp16'] && b.bundleNames?.['q8map-fp16']) out.push(b.bundleNames['q8map-fp16']);
          if (b.engines.w8a8 && b.bundleNames?.w8a8) out.push(b.bundleNames.w8a8);
          return out;
        });
        setTrtBundles(names);
      })
      .catch(() => {});
  };
  useEffect(() => { loadTrtBundles(); }, []);

  // Auto-select first available model when list loads and nothing is selected
  useEffect(() => {
    if (!models?.models) return;
    const dit = models.models.dit || [];
    const lm = models.models.lm || [];
    const vae = models.models.vae || [];
    const emb = models.models.embedding || [];

    if (dit.length > 0 && (!gp.ditModel || !dit.includes(gp.ditModel))) {
      gp.setDitModel(dit[0]);
    }
    if (lm.length > 0 && (!gp.lmModel || !lm.includes(gp.lmModel))) {
      gp.setLmModel(lm[0]);
    }
    if (vae.length > 0 && (!gp.vaeModel || !vae.includes(gp.vaeModel))) {
      gp.setVaeModel(vae[0]);
    }
    if (emb.length > 0 && (!gp.embeddingModel || !emb.includes(gp.embeddingModel))) {
      gp.setEmbeddingModel(emb[0]);
    }
  }, [models]);

  const baseDitModels = models?.models?.dit || [];
  // Append TRT bundles not already present in the engine's dit list.
  const ditModels = [...baseDitModels, ...trtBundles.filter(b => !baseDitModels.includes(b))];
  const lmModels = models?.models?.lm || [];
  const vaeModels = models?.models?.vae || [];
  const embeddingModels = models?.models?.embedding || [];

  return (
    <div className="space-y-3">
      {/* DiT Model */}
      <div>
        <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">{t('models.ditModel')}</label>
        <ModelSelect
          id="dit-model-select"
          value={gp.ditModel}
          onChange={gp.setDitModel}
          options={ditModels}
          formatLabel={formatDitModel}
          placeholder={t('common.loading')}
        />
        {getDitModelDescription(gp.ditModel) && (
          <p className="text-[10px] text-zinc-500 mt-1.5 leading-relaxed">{getDitModelDescription(gp.ditModel)}</p>
        )}
      </div>

      {/* LM Model */}
      <div>
        <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">{t('models.lmModel')}</label>
        <ModelSelect
          id="lm-model-select"
          value={gp.lmModel}
          onChange={gp.setLmModel}
          options={lmModels}
          formatLabel={formatLmModel}
          placeholder={t('common.loading')}
        />
        {getLmModelDescription(gp.lmModel) && (
          <p className="text-[10px] text-zinc-500 mt-1.5 leading-relaxed">{getLmModelDescription(gp.lmModel)}</p>
        )}
      </div>

      {/* VAE Model — only show when multiple VAEs are available */}
      {vaeModels.length > 1 && (
        <div>
          <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">{t('models.vaeDecoder')}</label>
          <ModelSelect
            id="vae-model-select"
            value={gp.vaeModel}
            onChange={gp.setVaeModel}
            options={vaeModels}
            formatLabel={formatVaeModel}
            placeholder={t('common.loading')}
          />
          {getVaeModelDescription(gp.vaeModel) && (
            <p className="text-[10px] text-zinc-500 mt-1.5 leading-relaxed">{getVaeModelDescription(gp.vaeModel)}</p>
          )}
        </div>
      )}

      {/* Text Encoder — only show when multiple are available */}
      {embeddingModels.length > 1 && (
        <div>
          <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">{t('models.textEncoder')}</label>
          <ModelSelect
            id="embedding-model-select"
            value={gp.embeddingModel}
            onChange={gp.setEmbeddingModel}
            options={embeddingModels}
            formatLabel={formatEmbeddingModel}
            placeholder={t('common.loading')}
          />
        </div>
      )}

      {/* Get More Models */}
      <div className="border-t border-zinc-200 dark:border-white/5 pt-3 mt-1">
        <button
          onClick={onModelManagerOpen}
          className="w-full px-3 py-2 rounded-xl bg-pink-500/10 border border-pink-500/20
                     text-sm text-pink-400 hover:bg-pink-500/20 hover:text-pink-300
                     transition-colors flex items-center justify-center gap-2"
        >
          <Download size={14} />
          {t('models.getMoreModels')}
        </button>
      </div>

      {/* Model Manager Modal — rendered by parent; shown here via showModelManager prop */}
      {showModelManager && (
        <ModelManagerModal onClose={() => {
          onModelManagerClose();
          sessionStorage.setItem('mm-auto-dismissed', '1');
          loadTrtBundles();
        }} />
      )}
    </div>
  );
};

/** Summary badge for the Models section */
export const ModelsBadge: React.FC = () => {
  const { ditModel, lmModel, vaeModel } = useGlobalParams();

  return (
    <span className="text-[10px] text-zinc-500 font-mono truncate">
      {formatDitModel(ditModel)} · {formatLmModel(lmModel)} · {formatVaeModel(vaeModel)}
    </span>
  );
};
