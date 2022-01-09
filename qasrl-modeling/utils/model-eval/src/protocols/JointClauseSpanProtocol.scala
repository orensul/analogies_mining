package qfirst.model.eval.protocols
import qfirst.model.eval._

import qfirst.clause.ext._

import cats.Id
import cats.Show
import cats.data.NonEmptyList
import cats.implicits._

import jjm.DependentMap
import jjm.LowerCaseString
import jjm.ling.ESpan
import jjm.ling.en.InflectedForms
import jjm.ling.en.VerbForm
import jjm.implicits._

import qasrl.labeling.SlotBasedLabel

case class JCSBeamItem(
  clause: String,
  clauseProb: Double,
  span: ESpan,
  spanProb: Double,
  answerSlots: Map[String, Double]
)

sealed trait JCSFilter extends Product with Serializable
case class JCSTripleFilter(
  clauseThreshold: Double,
  spanThreshold: Double,
  answerSlotThreshold: Double
) extends JCSFilter
case class JCSE2EFilter(
  threshold: Double
) extends JCSFilter
object JCSFilter {
  implicit def jcsFilterShow = Show.show[JCSFilter] {
    case JCSTripleFilter(c, s, as) =>
      f"c ≥ $c%.2f ∧ s ≥ $s%.2f ∧ slot <= $as%.2f"
    case JCSE2EFilter(t) =>
      f"p ≥ $t%.2f"
  }
}

case class JCSFilterSpace(
  clauseThresholds: List[Double],
  spanThresholds: List[Double],
  answerSlotThresholds: List[Double],
  e2eThresholds: List[Double])

object JointClauseSpanProtocol
    extends FactoringProtocol[
  JCSBeamItem, JCSFilter, JCSFilterSpace
] {
  override def getAllInnerFilters(fs: JCSFilterSpace): List[JCSFilter] = {
    val triples = for {
      c <- fs.clauseThresholds
      s <- fs.spanThresholds
      as <- fs.answerSlotThresholds
    } yield JCSTripleFilter(c, s, as)
    val e2es = fs.e2eThresholds.map(JCSE2EFilter(_))
    triples ++ e2es
  }

  def isModal(t: TAN) = t.tense match {
    case Tense.Finite.Modal(_) => true
    case _ => false
  }

  private[this] def constructQAs(
    clause: String,
    span: ESpan,
    answerSlot: String,
    beam: List[JCSBeamItem],
    getSpanTans: Option[ESpan => Set[TAN]],
    getAnimacy: Option[ESpan => Option[Boolean]]
  ): List[(SlotBasedLabel[VerbForm], ESpan)] = {
    val tans = getSpanTans.get(span).toList
    val animacy = getAnimacy.get

    if(tans.isEmpty) Nil else {
      val argToAnim = beam
        .filter(item => item.clause == clause)
        .flatMap(item => item.answerSlots.toList.map(item -> _))
        .groupBy { case (item, (answerSlot, answerSlotProb)) => answerSlot }
        .map { case (answerSlot, itemPairs) =>
          val anim = animacy(itemPairs.maxBy(p => p._2._2 * p._1.spanProb)._1.span).get
          answerSlot -> anim
        }

      val slotNames = "subj verb obj prep1 prep1-obj prep2 prep2-obj misc".split(" ").toList
      val clauseWords = clause.split(" ").toList
      val slotValues = clauseWords.take(7) ++ List(clauseWords.drop(7).mkString(" "))
      val slots = (slotNames, slotValues).zipped.toMap
      def getSlot(slot: String): Option[String] = Option(slots(slot)).filter(_ != "_")
      def getNoun(slot: String) = getSlot(slot).map(s => Noun.isAnimate.set(argToAnim(slot))(Noun.fromPlaceholder(s.lowerCase).get))
      def getMisc = getSlot("misc").map(s =>
        NonPrepArgument.nounLikeArgument
          .composePrism(NounLikeArgument.noun)
          .composeLens(Noun.isAnimate)
          .set(argToAnim("misc"))(NonPrepArgument.fromPlaceholder(s.lowerCase).get)
      )
      def getPrep(slot: String) = getSlot(slot).map(p => Preposition(p.lowerCase, getNoun(s"$slot-obj")))

      var argMap = DependentMap.empty[ArgumentSlot.Aux, Id]
      def putIfPresent[A](slot: ArgumentSlot.Aux[A], value: Option[A]) = {
        value.foreach { v =>
          argMap = argMap.put(slot, v)
        }
      }

      val isPassive = getSlot("verb").get == "verb[pss]"
      putIfPresent(Subj, getNoun("subj"))
      putIfPresent(Obj, getNoun("obj"))
      putIfPresent(Prep1, getPrep("prep1"))
      putIfPresent(Prep2, getPrep("prep2"))
      putIfPresent(Misc, getMisc)

      tans
        .filter(tan => !isPassive || !tan.isProgressive || (!tan.isPerfect && !isModal(tan)))
        .flatMap { tan =>
        val frame = Frame(
          ArgStructure(argMap, isPassive),
          InflectedForms.generic,
          tan)
        val questionStrings = frame.questionsForSlot(ArgumentSlot.fromString(answerSlot).get)
        questionStrings.headOption.map { questionString =>
          SlotBasedLabel.getVerbTenseAbstractedSlotsForQuestion(
            Vector(), InflectedForms.generic, List(questionString)
          ).head.get -> span
        }
      }
    }
  }

  override def getQAs(
    item: JCSBeamItem,
    beam: List[JCSBeamItem],
    filter: JCSFilter,
    getSpanTans: Option[ESpan => Set[TAN]],
    getAnimacy: Option[ESpan => Option[Boolean]]
  ): List[(SlotBasedLabel[VerbForm], ESpan)] = {
    filter match {
      case JCSTripleFilter(cThresh, sThresh, asThresh) =>
        for {
          clause <- List(item.clause)
          if item.clauseProb >= cThresh
          span <- List(item.span)
          if item.spanProb >= sThresh
          (answerSlot, asProb) <- item.answerSlots.toList
          if asProb >= asThresh
          qa <- constructQAs(clause, span, answerSlot, beam, getSpanTans, getAnimacy)
        } yield qa
      case JCSE2EFilter(thresh) =>
        for {
          clause <- List(item.clause)
          span <- List(item.span)
          (answerSlot, asProb) <- item.answerSlots.toList
          if (item.clauseProb * item.spanProb * asProb) >= thresh
          qa <- constructQAs(clause, span, answerSlot, beam, getSpanTans, getAnimacy)
        } yield qa
    }
  }
}
