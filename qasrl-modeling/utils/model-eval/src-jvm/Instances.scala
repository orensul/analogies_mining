package qfirst.model.eval

import cats.Functor
import cats.Monoid
import cats.Show
import cats.data.Ior
import cats.data.NonEmptyList
import cats.implicits._

import jjm.LowerCaseString
import jjm.ling.ESpan
import jjm.ling.Text
import jjm.ling.en.VerbForm
import jjm.ling.en.VerbForm.PastParticiple
import jjm.metrics._
import jjm.implicits._

// import qasrl.QuestionProcessor
// import qasrl.TemplateStateMachine

import qasrl.bank.Data
import qasrl.bank.AnnotationRound
import qasrl.bank.AnswerSource
import qasrl.bank.QuestionSource
// import qasrl.bank.FullData

import qasrl.data.Answer
import qasrl.data.AnswerLabel
import qasrl.data.Dataset
import qasrl.data.Sentence
import qasrl.data.VerbEntry
import qasrl.data.QuestionLabel

import qasrl.labeling.SlotBasedLabel

import HasMetrics.ops._

object Instances {

  case class SentenceInstance[A](
    gold: Sentence,
    pred: SentencePrediction[A]
  )

  case class VerbInstance[A](
    goldSentence: Sentence,
    gold: VerbEntry,
    pred: VerbPrediction[A]
  )

  def sentenceToVerbs[A] = (sentence: SentenceInstance[A]) => {
    // NOTE: filtering out "empty" verbs which are mistakenly present in the data
    sentence.gold.verbEntries.toList.filter(_._2.questionLabels.nonEmpty).flatMap {
      case (verbIndex, goldVerb) =>
        val predVerb = sentence.pred.verbs.find(_.verbIndex == verbIndex)
        if(predVerb.isEmpty) System.err.println("Could not find predicted verb: TODO better error message")
        predVerb.map(VerbInstance(sentence.gold, goldVerb, _))
    }: List[VerbInstance[A]]
  }

  case class QASetInstance(
    goldSentence: Sentence,
    goldVerb: VerbEntry,
    goldValid: Map[String, QuestionLabel],
    goldInvalid: Map[String, QuestionLabel],
    pred: Map[String, (SlotBasedLabel[VerbForm], Set[ESpan])]
  ) {
    def allQuestionStrings = (goldValid.keySet ++ goldInvalid.keySet ++ pred.keySet)
    def generalize = GeneralizedQASetInstance[SlotBasedLabel[VerbForm]](
      this,
      goldValid.map(p =>
        p._2.questionSlots -> p._2.answerJudgments.flatMap(_.judgment.getAnswer).flatMap(_.spans.toList).toSet
      ), pred.map(_._2)
    )
  }

  private def mapKeysAndAggregate[K0, K1, V: Monoid](m: Map[K0, V], f: K0 => K1) = {
    m.toList.foldMap { case (a, b) => Map(f(a) -> b) }
  }
  case class GeneralizedQASetInstance[A](
    original: QASetInstance,
    gold: Map[A, Set[ESpan]],
    pred: Map[A, Set[ESpan]]
  ) {
    def map[B](f: A => B): GeneralizedQASetInstance[B] = {
      GeneralizedQASetInstance(
        original,
        mapKeysAndAggregate(gold, f),
        mapKeysAndAggregate(pred, f),
      )
    }
    def getAcc = {
      pred.toList.foldMap { case (q, as) =>
        as.toList.foldMap { a =>
          val instance = (this, q, a)
          val isCorrect = gold.get(q).exists(_.contains(a))
          if(isCorrect) Accuracy.correct(instance) else Accuracy.incorrect(instance)
        }
      }
    }
    def stripAnswers = GeneralizedQASetInstance(
      original,
      gold.map { case (q, aSet) => q -> Set(ESpan(0, 1)) },
      pred.map { case (q, aSet) => q -> Set(ESpan(0, 1)) }
    )
  }

  val stripNegation = (slots: SlotBasedLabel[VerbForm]) => {
    def removeAuxNeg(aux: LowerCaseString) = (
      aux.toString match {
        case "can't" => "can"
        case "won't" => "will"
        case x if x.endsWith("n't") => x.dropRight(3)
        case x => x
      })
    def removeVerbNeg(verbPrefix: List[LowerCaseString]) = (
      verbPrefix.map(_.toString) match {
        case "not" :: rest => rest
        case x => x
      }
    ).map(_.lowerCase)

    import VerbForm._
    val res: SlotBasedLabel[VerbForm] = slots.aux.map(removeAuxNeg) match {
      case Some("did") if slots.subj.isEmpty =>
        slots.copy(aux = None, verbPrefix = removeVerbNeg(slots.verbPrefix), verb = Past)
      case Some("does") if slots.subj.isEmpty =>
        slots.copy(aux = None, verbPrefix = removeVerbNeg(slots.verbPrefix), verb = PresentSingular3rd)
      case Some(newAux) =>
        slots.copy(aux = Some(newAux.lowerCase), verbPrefix = removeVerbNeg(slots.verbPrefix))
      case None =>
        slots.copy(aux = None, verbPrefix = removeVerbNeg(slots.verbPrefix))
    }
    // if(slots != res) {
    //   println("vvvv")
    //   println(slots.renderQuestionString(InflectedForms.fromStrings("give", "gives", "giving", "gave", "given")))
    //   println(res.renderQuestionString(InflectedForms.fromStrings("give", "gives", "giving", "gave", "given")))
    //   println("^^^^")
    // }
    res
  }

  val stripTense = (slots: SlotBasedLabel[VerbForm]) => {
    val isNegated = slots.aux.exists(_.endsWith("n't".lowerCase)) || slots.verbPrefix.headOption.exists(_ == "not".lowerCase)
    val isPassive = slots.verb == PastParticiple &&
      (slots.aux.toList ++ slots.verbPrefix).map(_.toString).toSet.intersect(
        Set("be", "been", "is", "isn't", "was", "wasn't")
      ).nonEmpty
    slots.copy(aux = None, verbPrefix = Nil, verb = isNegated -> isPassive)
  }

  val stripAnimacy = (slots: SlotBasedLabel[VerbForm]) => {
    val stripArgAnim = (s: LowerCaseString) => s.toString
      .replaceAll("who", "what")
      .replaceAll("someone", "something")
      .lowerCase

    slots.copy(
      wh = stripArgAnim(slots.wh),
      subj = slots.subj.map(stripArgAnim),
      obj = slots.obj.map(stripArgAnim),
      obj2 = slots.obj2.map(stripArgAnim))
  }

  val stripTemplateWh = (template: TemplateSlots) => {
    template.copy(wh = "_".lowerCase)
  }

  val stripTemplatePrep = (template: TemplateSlots) => {
    template.prep.fold(template) { _ =>
      template.copy(prep = None, obj2 = None)
    }
  }

  val stripTemplateAll = (template: TemplateSlots) => {
    stripTemplateWh(stripTemplatePrep(template))
  }

  case class QuestionInstance(
    qas: QASetInstance,
    string: String,
    slots: SlotBasedLabel[VerbForm]
  )

  val qaSetToQuestions = (qas: QASetInstance) => {
    qas.allQuestionStrings.toList.map { qString =>
      val qSlots = qas.goldValid.get(qString).map(_.questionSlots).orElse {
        qas.goldInvalid.get(qString).map(_.questionSlots).orElse {
          qas.pred.get(qString).map(_._1)
        }
      }.get // should always work
      QuestionInstance(qas, qString, qSlots)
    }
  }

  val getQuestionBoundedAcc = (question: QuestionInstance) => {
    if(!question.qas.pred.contains(question.string)) BoundedAcc[QuestionInstance]()
    else if(question.qas.goldValid.contains(question.string)) BoundedAcc.correct(question)
    else if(question.qas.goldInvalid.contains(question.string)) BoundedAcc.incorrect(question)
    else BoundedAcc.uncertain(question)
  }

  val getQuestionConf = (question: QuestionInstance) => {
    val isPredicted = question.qas.pred.contains(question.string)
    val isTrue = isPredicted == question.qas.goldValid.contains(question.string)
    if(isTrue && isPredicted) BinaryConf.tp(question)
    else if(!isTrue && isPredicted) BinaryConf.fp(question)
    else if(!isTrue && !isPredicted) BinaryConf.fn(question)
    else BinaryConf.tn(question)
  }

  val getQuestionWithAnswerConf = (question: QuestionInstance) => {
    (question.qas.pred.get(question.string), question.qas.goldValid.get(question.string)) match {
      case (None, None) => BinaryConf.tn(question)
      case (Some(_), None) => BinaryConf.fp(question)
      case (None, Some(_)) => BinaryConf.fn(question)
      case (Some(predQA), Some(goldQA)) =>
        val predAnswerSpans = predQA._2
        val goldAnswerSpans = goldQA.answerJudgments.flatMap(_.judgment.getAnswer).flatMap(_.spans.toList).toSet
        if(predAnswerSpans.intersect(goldAnswerSpans).nonEmpty) {
          BinaryConf.tp(question)
        } else BinaryConf.fp(question) |+| BinaryConf.fn(question)
    }
  }

  val getQuestionWithAnswerBoundedAcc = (question: QuestionInstance) => {
    question.qas.pred.get(question.string).fold(BoundedAcc[QuestionInstance]()) { predQA =>
      question.qas.goldValid.get(question.string).map { validGoldQA =>
        val predAnswerSpans = predQA._2
        val goldAnswerSpans = validGoldQA.answerJudgments.flatMap(_.judgment.getAnswer).flatMap(_.spans.toList).toSet
        if(predAnswerSpans.intersect(goldAnswerSpans).nonEmpty) {
          BoundedAcc.correct(question)
        } else BoundedAcc.incorrect(question)
      }.orElse {
        question.qas.goldInvalid.get(question.string).as(BoundedAcc.incorrect(question))
      }.getOrElse(BoundedAcc.uncertain(question))
    }
  }

  case class QAInstance(
    question: QuestionInstance,
    span: ESpan
  )

  val questionToQAs = (question: QuestionInstance) => {
    val allSpans = question.qas.goldValid.get(question.string).toList.flatMap(_.answerJudgments.flatMap(_.judgment.getAnswer).flatMap(_.spans.toList)).toSet ++
      question.qas.goldInvalid.get(question.string).toList.flatMap(_.answerJudgments.flatMap(_.judgment.getAnswer).flatMap(_.spans.toList)).toSet ++
      question.qas.pred.get(question.string).toList.flatMap(_._2).toSet
    allSpans.toList.map(s => QAInstance(question, s))
  }

  val getQABoundedAcc = (qa: QAInstance) => {
    qa.question.qas.pred.get(qa.question.string).fold(BoundedAcc[QAInstance]()) { predQA =>
      if(!predQA._2.contains(qa.span)) { BoundedAcc[QAInstance]() }
      else {
        qa.question.qas.goldInvalid.get(qa.question.string)
          .as(BoundedAcc.incorrect(qa))
          .orElse {
          qa.question.qas.goldValid.get(qa.question.string).map { validGoldQA =>
            val goldAnswerSpans = validGoldQA.answerJudgments.flatMap(_.judgment.getAnswer).flatMap(_.spans.toList).toSet
            val goldContainsSpan = goldAnswerSpans.contains(qa.span)
            if(goldContainsSpan) BoundedAcc.correct(qa) else BoundedAcc.incorrect(qa)
          }
        }.getOrElse(BoundedAcc.uncertain(qa))
      }
    }
  }

  // gold invalid doesn't make sense: because invalid questions
  // may have been invalid due to tense issues or other things abstracted out by the template
  case class QATemplateSetInstance(
    qaSet: QASetInstance,
    gold: Map[String, (TemplateSlots, Set[ESpan])],
    pred: Map[String, (TemplateSlots, Set[ESpan])]
  )

  val qaSetToQATemplateSet = (qas: QASetInstance) => {
    val goldTemplates = qas.goldValid.values.toList.map { qLabel =>
      val templateSlots = TemplateSlots.fromQuestionSlots(qLabel.questionSlots)
      val answerSpans = qLabel.answerJudgments.flatMap(_.judgment.getAnswer).flatMap(_.spans.toList.toList).toSet
      templateSlots.toTemplateString -> (templateSlots, answerSpans)
    }.groupBy(_._1).map { case (string, templateSets) =>
      string -> (templateSets.head._2._1 -> templateSets.flatMap(_._2._2).toSet)
    }
    val predTemplates = qas.pred.values.map { case (slots, answerSpans) =>
      val templateSlots = TemplateSlots.fromQuestionSlots(slots)
      templateSlots.toTemplateString -> (templateSlots, answerSpans)
    }.groupBy(_._1).map { case (string, templateSets) =>
        string -> (templateSets.head._2._1, templateSets.flatMap(_._2._2).toSet)
    }
    QATemplateSetInstance(qas, goldTemplates, predTemplates)
  }

  case class QuestionTemplateInstance(
    qaTemplates: QATemplateSetInstance,
    string: String,
    slots: TemplateSlots
  )

  val qaTemplateSetToQuestionTemplates = (qaTemplates: QATemplateSetInstance) => {
    val allTemplateStrings = qaTemplates.gold.keySet ++ qaTemplates.pred.keySet
    allTemplateStrings.toList.map { tString =>
      val tSlots = qaTemplates.gold.get(tString).map(_._1).orElse {
        qaTemplates.pred.get(tString).map(_._1)
      }.get // should always work
      QuestionTemplateInstance(qaTemplates, tString, tSlots)
    }
  }

  val getQuestionTemplateConf = (template: QuestionTemplateInstance) => {
    val isPredicted = template.qaTemplates.pred.contains(template.string)
    val isTrue = isPredicted == template.qaTemplates.gold.contains(template.string)
    if(isTrue && isPredicted) BinaryConf.tp(template)
    else if(!isTrue && isPredicted) BinaryConf.fp(template)
    else if(!isTrue && !isPredicted) BinaryConf.fn(template)
    else BinaryConf.tn(template)
  }

  val getQuestionTemplateWithAnswerConf = (template: QuestionTemplateInstance) => {
    (template.qaTemplates.pred.get(template.string), template.qaTemplates.gold.get(template.string)) match {
      case (None, None) => BinaryConf.tn(template)
      case (Some(_), None) => BinaryConf.fp(template)
      case (None, Some(_)) => BinaryConf.fn(template)
      case (Some(predQA), Some(goldQA)) =>
        if(predQA._2.intersect(goldQA._2).nonEmpty) {
          BinaryConf.tp(template)
        } else BinaryConf.fp(template) |+| BinaryConf.fn(template)
    }
  }

  val getQuestionTemplateAcc = (template: QuestionTemplateInstance) => {
    if(!template.qaTemplates.pred.contains(template.string)) Accuracy[QuestionTemplateInstance]()
    else if(template.qaTemplates.gold.contains(template.string)) Accuracy.correct(template)
    else Accuracy.incorrect(template)
  }

  val getQuestionTemplateWithAnswerAcc = (template: QuestionTemplateInstance) => {
    (template.qaTemplates.pred.get(template.string), template.qaTemplates.gold.get(template.string)) match {
      case (None, None) => Accuracy[QuestionTemplateInstance]()
      case (Some(_), None) => Accuracy.incorrect(template)
      case (None, Some(_)) => Accuracy[QuestionTemplateInstance]()
      case (Some(predQA), Some(goldQA)) =>
        val predAnswerSpans = predQA._2
        val goldAnswerSpans = goldQA._2
        if(predAnswerSpans.intersect(goldAnswerSpans).nonEmpty) {
          Accuracy.correct(template)
        } else Accuracy.incorrect(template)
    }
  }

  case class QATemplateInstance(
    template: QuestionTemplateInstance,
    span: ESpan
  )

  val questionTemplateToQATemplates = (template: QuestionTemplateInstance) => {
    val allSpans = template.qaTemplates.gold.get(template.string).toList.flatMap(_._2).toSet ++
      template.qaTemplates.pred.get(template.string).toList.flatMap(_._2).toSet
    allSpans.toList.map(s => QATemplateInstance(template, s))
  }

  val getQATemplateAcc = (qa: QATemplateInstance) => {
    qa.template.qaTemplates.pred.get(qa.template.string).fold(Accuracy[QATemplateInstance]()) { predQA =>
      if(!predQA._2.contains(qa.span)) Accuracy[QATemplateInstance]() else {
        qa.template.qaTemplates.gold.get(qa.template.string).fold(Accuracy.incorrect(qa)) { goldQA =>
          if(goldQA._2.contains(qa.span)) Accuracy.correct(qa)
          else Accuracy.incorrect(qa)
        }
      }
    }
  }

  case class SpanSetInstance(
    qaSet: QASetInstance,
    gold: List[Set[ESpan]],
    pred: List[Set[ESpan]]
  )

  val qaSetToSpanSet = (qaSet: QASetInstance) => {
    val goldSpanSets = qaSet.goldValid.values.toList.map { qLabel =>
      qLabel.answerJudgments.toList
        .map(_.judgment)
        .flatMap(_.getAnswer)
        .map(_.spans.toList.toSet)
        .foldLeft(Set.empty[ESpan])(_ union _)
    }
    val predSpanSets = qaSet.pred.values.toList.map(_._2)
    SpanSetInstance(qaSet, goldSpanSets, predSpanSets)
  }

  // NOTE: does not do bipartite matching thing
  val getSpanSetConf = (spanSet: SpanSetInstance) => {
    case class SpanAlignment(
      remainingPred: Set[Set[ESpan]],
      conf: BinaryConf.Stats)
    val alignment = spanSet.gold.foldLeft(SpanAlignment(spanSet.pred.toSet, BinaryConf.Stats())) {
      case (SpanAlignment(preds, conf), goldSpanSet) =>
        preds.find(_.exists(s => goldSpanSet.exists(s.overlaps))) match {
          case None => (SpanAlignment(preds, conf |+| BinaryConf.Stats(fn = 1)))
          case Some(predSpanSet) => (SpanAlignment(preds - predSpanSet, conf |+| BinaryConf.Stats(tp = 1)))
        }
    }
    alignment.conf |+| BinaryConf.Stats(fp = alignment.remainingPred.size)
  }

  // left = predicted, right = gold
  case class AlignedSpanInstance(
    spanSet: SpanSetInstance,
    alignment: List[Ior[Set[ESpan], Set[ESpan]]],
    span: Ior[Set[ESpan], Set[ESpan]]
  )

  // NOTE: does not do bipartite matching thing
  val spanSetToAlignedSpans = (spanSet: SpanSetInstance) => {
    case class SpanAlignment(
      alignedSpans: List[Ior[Set[ESpan], Set[ESpan]]],
      remainingPred: Set[Set[ESpan]])
    val SpanAlignment(partialAlignment, unmatchedPreds) = spanSet.gold.foldLeft(SpanAlignment(Nil, spanSet.pred.toSet)) {
      case (SpanAlignment(alignment, preds), goldSpanSet) =>
        preds.find(_.exists(s => goldSpanSet.exists(s.overlaps))) match {
          case None => SpanAlignment(Ior.right(goldSpanSet) :: alignment, preds)
          case Some(predSpanSet) => SpanAlignment(Ior.both(predSpanSet, goldSpanSet) :: alignment, preds - predSpanSet)
        }
    }
    val alignment = unmatchedPreds.toList.map(Ior.left) ++ partialAlignment
    alignment.map(AlignedSpanInstance(spanSet, alignment, _))
  }

  val getAlignedSpanConf = (alignedSpan: AlignedSpanInstance) => {
    alignedSpan.span match {
      case Ior.Left(_) => BinaryConf.fp(alignedSpan)
      case Ior.Right(_) => BinaryConf.fn(alignedSpan)
      case Ior.Both(_, _) => BinaryConf.tp(alignedSpan)
    }
  }

  // def foldMapInstances[A: Monoid](
  //   gold: Dataset,
  //   pred: Map[String, SentencePrediction])(
  //   mapping: SentenceInstance => A
  // ): A = {
  //   gold.sentences.toList
  //     .filter(_._2.verbEntries.values.exists(_.questionLabels.nonEmpty)) // NOTE due to some "empty sentences" -- to fix in data
  //     .foldMap { case (sentenceId, goldSentence) =>
  //       pred.get(sentenceId) match {
  //         case None =>
  //           System.err.println("Could not find sentence; TODO better error message")
  //           Monoid[A].empty
  //         case Some(predSentence) =>
  //           mapping(SentenceInstance(goldSentence, predSentence))
  //       }
  //   }
  // }

  object Bucketers {

    def evalBucketBounds(
      unsortedBounds: NonEmptyList[Int])(
      value: Int
    ) = {
      val bounds = unsortedBounds.sorted
      if(value <= bounds.head) s"<=${bounds.head}"
      else bounds.toList.sliding(2).find(g => value > g(0) && value <= g(1)) match {
        case Some(g) => s"${g(0) + 1}-${g(1)}"
        case None => s">${bounds.last}"
      }
    }

    def sentenceLength(bounds: NonEmptyList[Int]) = (sent: Sentence) => {
      evalBucketBounds(bounds)(sent.sentenceTokens.size)
    }

    def verbFreq[A](
      getFreq: LowerCaseString => Int,
      bounds: NonEmptyList[Int]
    ) = (verb: VerbInstance[A]) => {
      val freq = getFreq(verb.gold.verbInflectedForms.stem)
      evalBucketBounds(bounds)(freq)
    }

    def goldDepLength(bounds: NonEmptyList[Int]) = (alignedSpan: AlignedSpanInstance) => {
      alignedSpan.span.right.fold("n/a")(
        _.flatMap(s => List(s.begin, s.end - 1))
          .map(_ - alignedSpan.spanSet.qaSet.goldVerb.verbIndex)
          .map(math.abs).min <| evalBucketBounds(bounds)
      )
    }

    def predDepLength(bounds: NonEmptyList[Int]) = (alignedSpan: AlignedSpanInstance) => {
      alignedSpan.span.left.fold("n/a")(
        _.flatMap(s => List(s.begin, s.end - 1))
          .map(_ - alignedSpan.spanSet.qaSet.goldVerb.verbIndex)
          .map(math.abs).min <| evalBucketBounds(bounds)
      )
    }

    val wh = (question: QuestionInstance) => {
      question.slots.wh.toString
    }

    val prep = (question: QuestionInstance) => {
      question.slots.prep.fold("_")(_.toString)
    }

    object Mappers {
      val prepIsPresent = (x: String) => if(x != "none") "yes" else "no"
      val whAdv = (x: String) => if(x == "who" || x == "what") "who/what" else "adv"
    }

    object Templated {
      val wh = (template: QuestionTemplateInstance) => {
        template.slots.wh.toString
      }
      val prep = (template: QuestionTemplateInstance) => {
        template.slots.prep.fold("_")(_.toString)
      }
    }
  }
}
