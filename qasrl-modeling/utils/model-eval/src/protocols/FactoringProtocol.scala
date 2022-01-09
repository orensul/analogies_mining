package qfirst.model.eval.protocols
import qfirst.model.eval._

import qfirst.clause.ext._

import cats.Show
import cats.implicits._

import jjm.LowerCaseString
import jjm.ling.ESpan
import jjm.ling.en.VerbForm
import jjm.implicits._

import qasrl.labeling.SlotBasedLabel

import io.circe.generic.JsonCodec

object FactoringProtocol {

  private def getTanFromString(s: String): TAN = {
    val tense = s.takeWhile(_ != ' ') match {
      case "past"    => Tense.Finite.Past
      case "present" => Tense.Finite.Present
      case m         => Tense.Finite.Modal(m.lowerCase)
    }
    TAN(
      tense = tense,
      isPerfect = s.contains("+pf"),
      isProgressive = s.contains("+prog"),
      isNegated = s.contains("+neg")
    )
  }

  private def makeTanMap(tanStringList: List[(String, Double)]): Map[TAN, Double] = {
    tanStringList.map { case (tan, prob) => getTanFromString(tan) -> prob }.toMap
  }

  @JsonCodec case class Beam[A](
    qa_beam: List[A],
    tans: Option[List[(String, Double)]],
    span_tans: Option[List[(ESpan, List[(String, Double)])]],
    animacy: Option[List[(ESpan, Double)]]) {
    val tanMap: Option[Map[TAN, Double]] = tans.map(makeTanMap)
    val spanTanMap: Option[Map[ESpan, Map[TAN, Double]]] = span_tans.map {
      pairList => pairList.map {
        case (span, tanStringList) => span -> makeTanMap(tanStringList)
      }.toMap
    }
    val animacyMap = animacy.map(_.toMap)
  }

  case class Filter[F](
    tanThreshold: Double,
    useSpanTans: Boolean,
    animacyNegativeThreshold: Double,
    animacyPositiveThreshold: Double,
    innerFilter: F)
  object Filter {
    def useSpanTansStr[F](f: Filter[F]) = if(f.useSpanTans) "∧ s -> t " else ""
    implicit def filterShow[F: Show] = Show.show[Filter[F]](f =>
      f"{ ${f.innerFilter.show}; t ≥ ${f.tanThreshold}%.2f ∧ a_n <= ${f.animacyNegativeThreshold}%.2f ∧ a_p ≥ ${f.animacyPositiveThreshold}%.2f ${useSpanTansStr(f)}}"
    )
  }

  case class FilterSpace[FS, F](
    tanThresholds: List[Double],
    useSpanTans: List[Boolean],
    animacyNegativeThresholds: List[Double],
    animacyPositiveThresholds: List[Double],
    innerSpace: FS,
    best: Option[Filter[F]])
}

import FactoringProtocol._

trait FactoringProtocol[A, F, FS] extends BeamProtocol[
  Beam[A], Filter[F], FilterSpace[FS, F]
] {

  def getAllInnerFilters(fs: FS): List[F]

  def getQAs(
    item: A,
    beam: List[A],
    filter: F,
    getSpanTans: Option[ESpan => Set[TAN]],
    getAnimacy: Option[ESpan => Option[Boolean]]
  ): List[(SlotBasedLabel[VerbForm], ESpan)]

  override def getAllFilters(fs: FilterSpace[FS, F]): List[Filter[F]] = {
    fs.best.map(List(_)).getOrElse(
      for {
        t <- fs.tanThresholds
        st <- fs.useSpanTans
        an <- fs.animacyNegativeThresholds
        ap <- fs.animacyPositiveThresholds
        if ap >= an
        f <- getAllInnerFilters(fs.innerSpace)
      } yield Filter(t, st, an, ap, f)
    )
  }

  def withBestFilter(fs: FilterSpace[FS, F], f: Option[Filter[F]]): FilterSpace[FS, F] = {
    fs.copy(best = f)
  }

  def filterBeam(filter: Filter[F], verb: VerbPrediction[Beam[A]]): Map[String, (SlotBasedLabel[VerbForm], Set[ESpan])] = {
    val beam = verb.beam
    val getSpanTans: Option[ESpan => Set[TAN]] = {
      if(filter.useSpanTans) {
        beam.spanTanMap.map { mapping =>
          (s: ESpan) => {
            mapping(s).toList.filter(_._2 >= filter.tanThreshold).map(_._1).toSet
          }
        }
      } else {
        beam.tanMap.map(mapping =>
          (s: ESpan) => {
            mapping.toList.filter(_._2 >= filter.tanThreshold).map(_._1).toSet
          }
        )
      }
    }
    val getAnimacy: Option[ESpan => Option[Boolean]] = {
      beam.animacyMap.map { animacyScores =>
        (s: ESpan) => {
          if(animacyScores(s) >= filter.animacyPositiveThreshold) Some(true)
          else if(animacyScores(s) < filter.animacyNegativeThreshold) Some(false)
          else None
        }
      }
    }
    verb.beam.qa_beam
      .flatMap(item => getQAs(item, verb.beam.qa_beam, filter.innerFilter, getSpanTans, getAnimacy))
      .groupBy(_._1.renderQuestionString(verb.verbInflectedForms.apply))
      .map { case (qString, qaItems) =>
        qString -> (qaItems.head._1 -> qaItems.map(_._2).toSet)
      }
  }
}
