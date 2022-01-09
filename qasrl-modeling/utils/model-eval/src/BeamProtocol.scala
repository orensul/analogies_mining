package qfirst.model.eval

import cats.Show

import jjm.ling.ESpan
import jjm.ling.en.InflectedForms
import jjm.ling.en.VerbForm

import qasrl.labeling.SlotBasedLabel

// law: bp.withBestFilter(fs, Some(f)).getAllFilters == List(f)
trait BeamProtocol[Beam, Filter, FilterSpace] {
  def getAllFilters(fs: FilterSpace): List[Filter]
  def withBestFilter(fs: FilterSpace, f: Option[Filter]): FilterSpace
  def filterBeam(filter: Filter, verb: VerbPrediction[Beam]): Map[String, (SlotBasedLabel[VerbForm], Set[ESpan])]
}
