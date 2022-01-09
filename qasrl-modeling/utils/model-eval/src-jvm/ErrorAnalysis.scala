package qfirst.model.eval


import jjm.LowerCaseString
import jjm.ling.ESpan
import jjm.metrics.Confusion
import jjm.implicits._

import cats.implicits._

object ErrorAnalysis {

  sealed trait PredClass
  object PredClass {
    case object NotPredicted extends PredClass
    case object Correct extends PredClass
    case object WrongAnswer extends PredClass
    case object CorrectTemplate extends PredClass
    case class WrongWh(pred: LowerCaseString, gold: LowerCaseString) extends PredClass
    case class SwappedPrep(pred: LowerCaseString, gold: LowerCaseString) extends PredClass
    case class MissingPrep(gold: LowerCaseString) extends PredClass
    case class ExtraPrep(pred: LowerCaseString) extends PredClass
    case object Other extends PredClass

    val notPredicted: PredClass = NotPredicted
    val correct: PredClass = Correct
    val wrongAnswer: PredClass = WrongAnswer
    val correctTemplate: PredClass = CorrectTemplate
    def wrongWh(pred: LowerCaseString, gold: LowerCaseString): PredClass = WrongWh(pred, gold)
    def swappedPrep(pred: LowerCaseString, gold: LowerCaseString): PredClass = SwappedPrep(pred, gold)
    def missingPrep(gold: LowerCaseString): PredClass = MissingPrep(gold)
    def extraPrep(pred: LowerCaseString): PredClass = ExtraPrep(pred)
    val other: PredClass = Other
  }

  def iouMatch(x: ESpan, y: ESpan) = {
    val xs = (x.begin until x.end).toSet
    val ys = (y.begin until y.end).toSet
    val i = xs.intersect(ys).size.toDouble
    val u = xs.union(ys).size.toDouble
    (i / u) >= 0.5
  }

  val computePredClass = (question: Instances.QuestionInstance) => {
    val P = PredClass
    question.qas.pred.get(question.string).fold(P.notPredicted) { case (predSlots, predSpans) =>
      question.qas.goldValid.get(question.string) match {
        case Some(qLabel) =>
          val answerSpans = qLabel.answerJudgments.flatMap(_.judgment.getAnswer).flatMap(_.spans.toList).toSet
          if(answerSpans.exists(g => predSpans.exists(p => iouMatch(g, p)))) P.correct else P.wrongAnswer
        case None =>
          val qaTemplates = Instances.qaSetToQATemplateSet(question.qas)
          val predTemplateSlots = TemplateSlots.fromQuestionSlots(predSlots)
          qaTemplates.gold.get(predTemplateSlots.toTemplateString).as(P.correctTemplate).getOrElse {
            def abstractWh(slots: TemplateSlots) = slots.copy(wh = (if(slots.wh.toString == "what") "what" else "adv").lowerCase)
            val whAbstractedPredTemplateSlots = abstractWh(predTemplateSlots)
            val whError = qaTemplates.gold.values.toList.map(_._1).filter { goldSlots =>
              abstractWh(goldSlots) == whAbstractedPredTemplateSlots
            }.map(goldSlots => P.wrongWh(predTemplateSlots.wh, goldSlots.wh)).headOption
            whError.getOrElse {
              val prepError = predTemplateSlots.prep match {
                case None => // missing prep
                  qaTemplates.gold.values.toList.map(_._1).filter { goldSlots =>
                    abstractWh(goldSlots).copy(prep = None) == whAbstractedPredTemplateSlots ||
                      abstractWh(goldSlots).copy(prep = None, obj2 = None) == whAbstractedPredTemplateSlots
                  }.flatMap(_.prep.map(P.missingPrep(_))).headOption
                case Some(predPrep) =>
                  qaTemplates.gold.values.toList.map(_._1).filter { goldSlots =>
                    abstractWh(goldSlots).copy(prep = Some(predPrep)) == whAbstractedPredTemplateSlots
                  }.flatMap(_.prep.map(P.swappedPrep(predPrep, _))).headOption.orElse {
                    qaTemplates.gold.values.toList.map(_._1).filter { goldSlots =>
                      whAbstractedPredTemplateSlots.copy(prep = None) == abstractWh(goldSlots) ||
                        whAbstractedPredTemplateSlots.copy(prep = None, obj2 = None) == abstractWh(goldSlots)
                    }.headOption.as(P.extraPrep(predPrep))
                  }
              }
              prepError.getOrElse(P.other)
            }
          }
      }
    }
  }

  def runPrepositionAnalysis(predClasses: List[(PredClass, Instances.QuestionInstance)]) = {
    val prepConf = {
      import PredClass._
      predClasses.collect {
        case (SwappedPrep(pred, gold), q) => Confusion.instance(gold, pred, q)
        case (Correct | WrongAnswer | CorrectTemplate | WrongWh(_, _), q) =>
          val prep = q.slots.prep.getOrElse("_".lowerCase)
          Confusion.instance(prep, prep, q)
        case (MissingPrep(gold), q) => Confusion.instance(gold, "_".lowerCase, q)
        case (ExtraPrep(pred), q) => Confusion.instance("_".lowerCase, pred, q)
      }.combineAll
    }

    println(prepConf.stats.prettyString(10))

    val allConfusionPairs = prepConf.matrix.toList.flatMap {
      case (gold, predMap) => predMap.toList.map {
        case (pred, questions) => (gold, pred, questions)
      }
    }
    allConfusionPairs
      .filter(t => t._1 != t._2)
      .sortBy(-_._3.size).takeWhile(_._3.size >= 10).foreach {
      case (gold, pred, questions) =>
        val verbHist = questions.groupBy(_.qas.goldVerb.verbInflectedForms.stem).map {
          case (verb, qs) => verb -> qs.size
        }
        val numInstances = questions.size
        println(s"Gold: $gold; Pred: $pred; num confusions: $numInstances")
        verbHist.toList
          .sortBy(-_._2)
          .takeWhile(_._2 > 1)
          .takeWhile(_._2.toDouble / numInstances >= 0.04)
          .foreach { case (verb, num) =>
            println(f"$verb%12s $num%3d (${num * 100.0 / numInstances}%4.1f%%)")
        }
    }

    val prepPresenceInstances = for {
      (gold, pred, questions) <- allConfusionPairs
      if gold != pred && gold.toString != "_" && pred.toString != "_"
      question <- questions
    } yield {
      val tokens = question.qas.goldSentence.sentenceTokens
      val goldInSentence = tokens.contains(gold.toString)
      val predInSentence = tokens.contains(pred.toString)
      val label = (goldInSentence, predInSentence) match {
        case (true,   true) => "both in sentence"
        case (false,  true) => "pred in sentence"
        case (true,  false) => "gold in sentence"
        case (false, false) => "neither in sentence"
      }
      label -> (gold, pred)
    }

    val prepPresenceCounts = prepPresenceInstances
      .groupBy(_._1).map { case (k, vs) => k -> vs.map(_._2) }

    prepPresenceCounts.foreach { case (k, goldPredPairs) =>
      println(f"$k%20s: ${goldPredPairs.size}%d (${goldPredPairs.size * 100.0 / prepPresenceInstances.size}%3.1f%%)")
      goldPredPairs.groupBy(identity).map { case (k, vs) => k -> vs.size }.toList.sortBy(-_._2).take(5).foreach {
        case ((gold, pred), num) => println(f"Gold: $gold%10s | Pred: $pred%10s $num%d")
      }
    }
  }
}
