import React, { useState } from 'react'
import { ArrowRightLeft, CheckCircle2, XCircle, Loader2 } from 'lucide-react'
import { Button } from '../ui/button'
import { SectionCard } from '../layout/SectionCard'
import { Badge } from '../ui/badge'
import { alphaApi } from '../../utils/alphaApi'
import { toast } from 'sonner'

interface MigrationResult {
  original_formula: string
  migrated_formula: string | null
  source_market: string
  target_market: string
  field_mappings: Record<string, string>
  unmappable_fields: string[]
  is_viable: boolean
}

export function MigrationTab() {
  const [formula, setFormula] = useState('')
  const [source, setSource] = useState('crypto')
  const [target, setTarget] = useState('a_share')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<MigrationResult | null>(null)

  const handleMigrate = async () => {
    if (!formula.trim()) return
    setLoading(true)
    setResult(null)
    try {
      const resp = await alphaApi.migrateFormula({
        formula: formula.trim(),
        source_market: source,
        target_market: target,
      })
      setResult(resp)
      if (resp.is_viable) {
        toast.success('Factor migration successful')
      } else {
        toast.warning(`Migration partial: ${resp.unmappable_fields.length} unmappable fields`)
      }
    } catch (err) {
      toast.error('Migration failed')
    } finally {
      setLoading(false)
    }
  }

  const swap = () => {
    setSource(target)
    setTarget(source)
    setResult(null)
  }

  return (
    <div className="space-y-5">
      <SectionCard title="Cross-market Migration" description="将因子公式从一个市场迁移到另一个市场">
        <div className="space-y-4">
          <div>
            <label className="text-xs font-medium text-muted-foreground">Formula</label>
            <textarea
              className="glass-input mt-1 w-full font-mono text-sm"
              rows={2}
              placeholder="ts_mean(close, 20) / ts_std(close, 20)"
              value={formula}
              onChange={(e) => setFormula(e.target.value)}
            />
          </div>

          <div className="flex items-center gap-3">
            <div className="flex-1">
              <label className="text-xs text-muted-foreground">Source</label>
              <select className="glass-input mt-1" value={source} onChange={(e) => setSource(e.target.value)}>
                <option value="crypto">Crypto</option>
                <option value="a_share">A-Share</option>
              </select>
            </div>

            <button
              onClick={swap}
              className="mt-5 rounded-full p-2 text-muted-foreground transition hover:bg-secondary hover:text-foreground"
            >
              <ArrowRightLeft size={16} />
            </button>

            <div className="flex-1">
              <label className="text-xs text-muted-foreground">Target</label>
              <select className="glass-input mt-1" value={target} onChange={(e) => setTarget(e.target.value)}>
                <option value="crypto">Crypto</option>
                <option value="a_share">A-Share</option>
              </select>
            </div>
          </div>

          <Button onClick={handleMigrate} disabled={loading || !formula.trim()} className="gap-2">
            {loading ? <Loader2 size={14} className="animate-spin" /> : <ArrowRightLeft size={14} />}
            Migrate
          </Button>
        </div>
      </SectionCard>

      {result && (
        <SectionCard
          title="Migration Result"
          action={
            result.is_viable ? (
              <Badge variant="outline" className="border-emerald-500/50 text-emerald-500">Viable</Badge>
            ) : (
              <Badge variant="outline" className="border-red-500/50 text-red-500">Partial</Badge>
            )
          }
        >
          <div className="space-y-4">
            {result.migrated_formula && (
              <div>
                <div className="text-xs text-muted-foreground mb-1">Migrated Formula</div>
                <code className="block rounded-xl bg-secondary/60 px-4 py-3 font-mono text-sm break-all">
                  {result.migrated_formula}
                </code>
              </div>
            )}

            <div>
              <div className="text-xs text-muted-foreground mb-2">Field Mappings</div>
              <div className="grid gap-1.5">
                {Object.entries(result.field_mappings).map(([src, tgt]) => (
                  <div key={src} className="flex items-center gap-2 text-sm">
                    <CheckCircle2 size={13} className="text-emerald-500 shrink-0" />
                    <code className="text-xs">{src}</code>
                    <span className="text-muted-foreground">→</span>
                    <code className="text-xs">{tgt}</code>
                  </div>
                ))}
                {result.unmappable_fields.map((f) => (
                  <div key={f} className="flex items-center gap-2 text-sm">
                    <XCircle size={13} className="text-red-500 shrink-0" />
                    <code className="text-xs">{f}</code>
                    <span className="text-xs text-muted-foreground">— no equivalent in {result.target_market}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </SectionCard>
      )}
    </div>
  )
}
