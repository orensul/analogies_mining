package qfirst.model.eval.protocols
import qfirst.model.eval._

import cats.Show
import cats.implicits._

import jjm.ling.ESpan
import jjm.ling.en.VerbForm

import qasrl.labeling.SlotBasedLabel

object QAsWithInv {
  case class BeamItem(
    questionSlots: SlotBasedLabel[VerbForm],
    questionProb: Double,
    invalidProb: Double,
    span: ESpan,
    spanProb: Double)

  case class Filter(
    questionThreshold: Double,
    invalidThreshold: Double,
    spanThreshold: Double,
    spanBeatsInvalid: Boolean)
  object Filter {
    def spanBeatsInvalidStr(f: Filter) = if(f.spanBeatsInvalid) "s ≥ i " else ""
    implicit val filterShow = Show.show[Filter](f =>
      f"{ q ≥ ${f.questionThreshold}%.2f ∧ s ≥ ${f.spanThreshold}%.2f ∧ i <= ${f.invalidThreshold}%.2f ${spanBeatsInvalidStr(f)}}"
    )
  }

  case class FilterSpace(
    questionThresholds: List[Double],
    invalidThresholds: List[Double],
    spanThresholds: List[Double],
    best: Option[Filter])

  val protocol = new BeamProtocol[List[BeamItem], Filter, FilterSpace] {
    def getAllFilters(fs: FilterSpace): List[Filter] = {
      fs.best.fold(
        for {
          q <- fs.questionThresholds
          i <- fs.invalidThresholds
          s <- fs.spanThresholds
          sbi <- List(true, false)
        } yield Filter(q, i, s, sbi)
      )(List(_))
    }
    def withBestFilter(fs: FilterSpace, f: Option[Filter]): FilterSpace = {
      fs.copy(best = f)
    }
    def filterBeam(filter: Filter, verb: VerbPrediction[List[BeamItem]]): Map[String, (SlotBasedLabel[VerbForm], Set[ESpan])] = {
      verb.beam
        .filter(item =>
          item.questionProb >= filter.questionThreshold &&
            item.invalidProb <= filter.invalidThreshold &&
            item.spanProb >= filter.spanThreshold &&
            (!filter.spanBeatsInvalid || item.spanProb >= item.invalidProb))
        .groupBy(_.questionSlots.renderQuestionString(verb.verbInflectedForms.apply))
        .map { case (qString, qaItems) =>
          qString -> (qaItems.head.questionSlots -> qaItems.map(_.span).toSet)
        }
    }
    // use this to mimic the old A-first decoding method
    private[this] def filterBeamAfirstOld(filter: Filter, verb: VerbPrediction[List[BeamItem]]): Map[String, (SlotBasedLabel[VerbForm], Set[ESpan])] = {
      verb.beam.groupBy(i => i.span -> i.spanProb)
        .filter(_._1._2 >= filter.spanThreshold)
        .toList.sortBy(-_._1._2)
        .foldLeft(List.empty[(String, (SlotBasedLabel[VerbForm], ESpan))]) {
          case (acc, ((span, spanProb), items)) =>
            if(!acc.map(_._2._2).exists(span.overlaps)) {
              val qSlots = items.maxBy(_.questionProb).questionSlots
              val qString = qSlots.renderQuestionString(verb.verbInflectedForms.apply)
              (qString -> (qSlots -> span)) :: acc
            } else acc
        }.groupBy(_._1)
        .map { case (qString, tuples) =>
          qString -> (tuples.head._2._1 -> tuples.map(_._2._2).toSet)
        }
    }
    // use this to mimic the old Q-first decoding method
    private[this] def filterBeamQfirstOld(filter: Filter, verb: VerbPrediction[List[BeamItem]]): Map[String, (SlotBasedLabel[VerbForm], Set[ESpan])] = {
      verb.beam.groupBy(i => i.questionSlots -> i.questionProb)
        .filter(_._1._2 >= filter.questionThreshold)
        .toList.sortBy(-_._1._2)
        .foldLeft(List.empty[(String, (SlotBasedLabel[VerbForm], Set[ESpan]))]) {
          case (acc, ((qSlots, qProb), items)) =>
            val answers = items
              .filter(_.spanProb >= filter.spanThreshold)
              .map(_.span)
              .foldLeft(Set.empty[ESpan]) { (spanSet, span) =>
                if(!spanSet.exists(span.overlaps) && !acc.flatMap(_._2._2).toSet.exists(span.overlaps)) {
                  spanSet + span
                } else spanSet
              }
            if(answers.isEmpty) acc else {
              val qString = qSlots.renderQuestionString(verb.verbInflectedForms.apply)
              (qString -> (qSlots -> answers)) :: acc
            }
        }.toMap
    }
  }
}
