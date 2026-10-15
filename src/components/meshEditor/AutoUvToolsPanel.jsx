// Auto UV mode left panel. Exposes every autouv.unwrap() parameter, runs the
// unwrap via the Python mesh-tools service, and offers Keep/Revert on the result.
// Presentational: option state + handlers come from MeshEditorPage.
import { RangeField, NumberField, ToggleField, SelectField } from './MeshToolField'
import MeshToolResult from './MeshToolResult'
import MeshToolProgress from './MeshToolProgress'

export default function AutoUvToolsPanel({
  options,
  setOption,
  running,
  result,
  progress,
  onRun,
  onKeepResult,
  onRevertResult,
  disabled,
}) {
  const o = options
  const fieldsDisabled = disabled || running

  return (
    <>{/* AUTO UV */}
      <div className="mesh-editor-panel__section">
        <span className="mesh-editor-panel__section-title">Auto UV</span>
        <button
          type="button"
          className="mesh-editor-btn mesh-editor-btn--primary"
          onClick={onRun}
          disabled={disabled || running}
          title="Unwrap UVs with the Python service"
        >
          <span className="material-symbols-outlined">{running ? 'progress_activity' : 'auto_awesome'}</span>
          <span>{running ? 'Unwrapping…' : 'Run Auto UV'}</span>
        </button>

        {running && <MeshToolProgress progress={progress} />}

        {result && (
          <MeshToolResult
            title="UV unwrap applied"
            rows={result.rows}
            previewUrl={result.previewUrl}
            previewAlt="UV layout"
            onKeep={onKeepResult}
            onRevert={onRevertResult}
            disabled={running}
          />
        )}
      </div>

      <div className="mesh-editor-panel__section">
        <span className="mesh-editor-panel__section-title">Segmentation</span>
        <RangeField label="Normal-cone cap" suffix="°" min={1} max={180} step={1}
          value={o.max_cone_deg} onChange={v => setOption('max_cone_deg', v)} disabled={fieldsDisabled}
          hint="Higher = fewer, more distorted charts" />
        <RangeField label="Sharp-edge weight" min={0} max={1} step={0.01} decimals={2}
          value={o.sharp_weight} onChange={v => setOption('sharp_weight', v)} disabled={fieldsDisabled}
          hint="How strongly sharp edges attract seams" />
        <RangeField label="Fold cap" suffix="°" min={1} max={180} step={1}
          value={o.fold_cap_deg} onChange={v => setOption('fold_cap_deg', v)} disabled={fieldsDisabled}
          hint="Dihedral fold angle that forces a seam" />
        <NumberField label="Min faces / chart" min={1} step={1}
          value={o.min_faces} onChange={v => setOption('min_faces', v)} disabled={fieldsDisabled}
          hint="Charts smaller than this are dissolved into neighbours" />
        <NumberField label="Min area fraction" min={0} max={1} step={0.001}
          value={o.min_area_frac} onChange={v => setOption('min_area_frac', v)} disabled={fieldsDisabled}
          hint="Min chart area as a fraction of total surface area" />
      </div>

      <div className="mesh-editor-panel__section">
        <span className="mesh-editor-panel__section-title">Refinement</span>
        <ToggleField label="Validated merge pass" value={o.refine}
          onChange={v => setOption('refine', v)} disabled={fieldsDisabled}
          hint="LSCM-validated chart merge (off = faster, more charts)" />
        <NumberField label="Merge below faces" min={1} step={1}
          value={o.refine_target_faces} onChange={v => setOption('refine_target_faces', v)}
          disabled={fieldsDisabled || !o.refine}
          hint="Charts below this face count are merge candidates" />
        <RangeField label="Merge distortion cap" min={1} max={10} step={0.01} decimals={2}
          value={o.refine_ad_thresh} onChange={v => setOption('refine_ad_thresh', v)}
          disabled={fieldsDisabled || !o.refine}
          hint="Max angle-distortion ratio a merge may introduce" />
      </div>

      <div className="mesh-editor-panel__section">
        <span className="mesh-editor-panel__section-title">Seam placement</span>
        <ToggleField label="Hide seams" value={o.hide_seams}
          onChange={v => setOption('hide_seams', v)} disabled={fieldsDisabled}
          hint="Price seams by visibility (occlusion + concavity) so they move into creases and hidden places" />
        <RangeField label="Hide strength" min={0} max={4} step={0.1} decimals={1}
          value={o.hide_strength} onChange={v => setOption('hide_strength', v)}
          disabled={fieldsDisabled || !o.hide_seams}
          hint="How much more a seam costs on open surface than in a hidden crease" />
        <ToggleField label="Move borders to creases" value={o.refine_borders}
          onChange={v => setOption('refine_borders', v)} disabled={fieldsDisabled}
          hint="Re-route every chart border along the cheapest nearby path: shorter, straighter, on sharp or hidden edges" />
        <RangeField label="Border reach" suffix=" rings" min={1} max={16} step={1}
          value={o.border_rings} onChange={v => setOption('border_rings', v)}
          disabled={fieldsDisabled || !o.refine_borders}
          hint="How many face rings either side of a border it may move" />
        <ToggleField label="Straighten seams" value={o.straighten_seams}
          onChange={v => setOption('straighten_seams', v)} disabled={fieldsDisabled}
          hint="Re-route chart borders with a cost on turning. Helps on quad-based meshes; little effect on triangulated AI meshes" />
        <ToggleField label="Character layout" value={o.character_rescue}
          onChange={v => setOption('character_rescue', v)} disabled={fieldsDisabled}
          hint="On organic models that come out sheared or fragmented, also try cutting limbs at the joints and front from back, and keep the better result" />
        <ToggleField label="Seam on mirror plane" value={o.symmetry}
          onChange={v => setOption('symmetry', v)} disabled={fieldsDisabled}
          hint="Find the mirror plane and keep a seam down it (needs mirrored triangles, e.g. symmetric retopology)" />
      </div>

      <div className="mesh-editor-panel__section">
        <span className="mesh-editor-panel__section-title">Parameterization</span>
        <SelectField label="Method" value={o.method} onChange={v => setOption('method', v)} disabled={fieldsDisabled}
          options={[
            { value: 'auto', label: 'Auto' },
            { value: 'lscm', label: 'LSCM' },
            { value: 'arap', label: 'ARAP' },
            { value: 'planar', label: 'Planar' },
          ]} />
        <RangeField label="ARAP iterations" min={0} max={100} step={1}
          value={o.arap_iters} onChange={v => setOption('arap_iters', v)} disabled={fieldsDisabled}
          hint="0 disables ARAP (LSCM/planar only)" />
        <ToggleField label="Re-join islands" value={o.rejoin}
          onChange={v => setOption('rejoin', v)} disabled={fieldsDisabled}
          hint="After flattening, merge neighbouring islands wherever the union still flattens cleanly: fewer islands and seams, a little more distortion" />
        <RangeField label="Join strength" min={1.05} max={3} step={0.05} decimals={2}
          value={o.rejoin_limit} onChange={v => setOption('rejoin_limit', v)}
          disabled={fieldsDisabled || !o.rejoin}
          hint="How much shear a re-joined island may reach. Higher = fewer islands, more distortion" />
        <ToggleField label="Relax stretched islands" value={o.relax_creases}
          onChange={v => setOption('relax_creases', v)} disabled={fieldsDisabled}
          hint="Open the sharp creases of badly stretched islands, or split smooth ones in two" />
        <ToggleField label="Absorb strip islands" value={o.absorb_ribbons}
          onChange={v => setOption('absorb_ribbons', v)} disabled={fieldsDisabled}
          hint="Lay thin strips (chamfers, cuffs) back onto the island they border" />
        <ToggleField label="Unroll bands" value={o.unroll_bands}
          onChange={v => setOption('unroll_bands', v)} disabled={fieldsDisabled}
          hint="Lay treads, pipes and cans out as exact rectangles" />
        <ToggleField label="Cut non-disk charts" value={o.ensure_disks}
          onChange={v => setOption('ensure_disks', v)} disabled={fieldsDisabled}
          hint="Open tubes, closed shells and handled charts with the cheapest cut so they flatten without folding" />
      </div>

      <div className="mesh-editor-panel__section">
        <span className="mesh-editor-panel__section-title">Packing</span>
        <SelectField label="Atlas resolution" value={String(o.resolution)}
          onChange={v => setOption('resolution', Number(v))} disabled={fieldsDisabled}
          options={[256, 512, 1024, 2048, 4096, 8192].map(n => ({ value: String(n), label: `${n} px` }))} />
        <RangeField label="Padding" suffix=" texels" min={0} max={64} step={1}
          value={o.padding_texels} onChange={v => setOption('padding_texels', v)} disabled={fieldsDisabled} />
        <ToggleField label="Stack repeats" value={o.stack_repeats}
          onChange={v => setOption('stack_repeats', v)} disabled={fieldsDisabled}
          hint="Rigid copies and rotational repeats share their master's UVs. Stacked copies cannot be painted differently" />
        <SelectField label="World up" value={o.orient_up} onChange={v => setOption('orient_up', v)} disabled={fieldsDisabled}
          hint="Turn islands so this world axis points up the texture (tiling bark, brick, rust). Islands then pack unrotated"
          options={[
            { value: 'off', label: 'Off' },
            { value: 'y', label: 'Y (glTF up)' },
            { value: 'z', label: 'Z' },
            { value: 'x', label: 'X' },
          ]} />
      </div>

      <div className="mesh-editor-panel__section">
        <span className="mesh-editor-panel__section-title">Topology repair</span>
        <ToggleField label="Proximity weld" value={o.weld}
          onChange={v => setOption('weld', v)} disabled={fieldsDisabled}
          hint="Weld coincident verts before unwrapping (stitches shattered shells)" />
        <RangeField label="Weld tolerance" min={0} max={1} step={0.01} decimals={2}
          value={o.weld_tol_frac} onChange={v => setOption('weld_tol_frac', v)}
          disabled={fieldsDisabled || !o.weld}
          hint="As a fraction of median edge length" />
      </div>

      <div className="mesh-editor-panel__section">
        <span className="mesh-editor-panel__section-title">Shading</span>
        <ToggleField label="Preserve normals" value={o.preserve_normals}
          onChange={v => setOption('preserve_normals', v)} disabled={fieldsDisabled}
          hint="Carry the mesh's own vertex normals through the unwrap, so shading is unchanged" />
        <RangeField label="Smoothing angle" suffix="°" min={0} max={180} step={1}
          value={o.normal_smooth_deg} onChange={v => setOption('normal_smooth_deg', v)}
          disabled={fieldsDisabled || o.preserve_normals}
          hint="Used when normals are rebuilt: edges sharper than this stay hard (180 = fully smooth)" />
      </div>

      <div className="mesh-editor-panel__notes">
        <span className="mesh-editor-panel__hint">Auto UV runs on the Python mesh-tools service (Settings → Mesh Tools).</span>
        <span className="mesh-editor-panel__hint">The result replaces the mesh; use Keep or Revert to decide.</span>
      </div>
    </>
  )
}
