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

import io.circe.generic.JsonCodec

@JsonCodec case class QfirstBeamItem[Q](
  questionSlots: Q,
  questionProb: Double,
  invalidProb: Double,
  span: ESpan,
  spanProb: Double) {
  def render(renderQ: Q => String) = {
    f"q: $questionProb%4.2f s: $spanProb%4.2f i: $invalidProb%4.2f   ${renderQ(questionSlots)}%80s     $span"
  }
}

case class QfirstFilter(
  questionThreshold: Double,
  invalidThreshold: Double,
  spanThreshold: Double,
  spanBeatsInvalid: Boolean,
  spanAnimacyThreshold: Double)
object QfirstFilter {
  def spanBeatsInvalidStr(f: QfirstFilter) = if(f.spanBeatsInvalid) "∧ s ≥ i " else ""
  implicit val filterShow = Show.show[QfirstFilter](f =>
    f"q ≥ ${f.questionThreshold}%.2f ∧ s ≥ ${f.spanThreshold}%.2f ∧ i <= ${f.invalidThreshold}%.2f ${spanBeatsInvalidStr(f)}%s∧ s_a ≥ ${f.spanAnimacyThreshold}%.2f"
  )
}

case class QfirstFilterSpace(
  questionThresholds: List[Double],
  invalidThresholds: List[Double],
  spanThresholds: List[Double],
  spanBeatsInvalid: List[Boolean] = List(true, false),
  spanAnimacyThresholds: List[Double])

trait QfirstProtocol[Q] extends FactoringProtocol[
  QfirstBeamItem[Q], QfirstFilter, QfirstFilterSpace] {

  def getQuestions(
    question: Q,
    span: ESpan,
    beam: List[QfirstBeamItem[Q]],
    getSpanTans: Option[ESpan => Set[TAN]],
    getAnimacy: Option[ESpan => Option[Boolean]],
    spanAnimacyThreshold: Double
  ): List[SlotBasedLabel[VerbForm]]

  override def getAllInnerFilters(fs: QfirstFilterSpace): List[QfirstFilter] = {
    for {
      q <- fs.questionThresholds
      i <- fs.invalidThresholds
      s <- fs.spanThresholds
      sbi <- fs.spanBeatsInvalid
      sat <- fs.spanAnimacyThresholds
    } yield QfirstFilter(q, i, s, sbi, sat)
  }

  override def getQAs(
    item: QfirstBeamItem[Q],
    beam: List[QfirstBeamItem[Q]],
    filter: QfirstFilter,
    getSpanTans: Option[ESpan => Set[TAN]],
    getAnimacy: Option[ESpan => Option[Boolean]]
  ): List[(SlotBasedLabel[VerbForm], ESpan)] = {
    for {
      span <- List(item.span)
      if item.spanProb >= filter.spanThreshold
      if !filter.spanBeatsInvalid || item.spanProb >= item.invalidProb
      if item.questionProb >= filter.questionThreshold
      if item.invalidProb < filter.invalidThreshold
      slots <- getQuestions(item.questionSlots, span, beam, getSpanTans, getAnimacy, filter.spanAnimacyThreshold)
      fixedSlots <- QuestionSlotFixer(slots)
    } yield (fixedSlots, span)
  }

}

object QfirstFullProtocol extends QfirstProtocol[SlotBasedLabel[VerbForm]] {
  def getQuestions(
    question: SlotBasedLabel[VerbForm],
    span: ESpan,
    beam: List[QfirstBeamItem[SlotBasedLabel[VerbForm]]],
    getSpanTans: Option[ESpan => Set[TAN]],
    getAnimacy: Option[ESpan => Option[Boolean]],
    spanAnimacyThreshold: Double
  ): List[SlotBasedLabel[VerbForm]] =
    List(question)
}

object QfirstNoTanProtocol extends QfirstProtocol[Map[String, String]] {
  def getQuestions(
    question: Map[String, String],
    span: ESpan,
    beam: List[QfirstBeamItem[Map[String, String]]],
    getSpanTans: Option[ESpan => Set[TAN]],
    getAnimacy: Option[ESpan => Option[Boolean]],
    spanAnimacyThreshold: Double
  ): List[SlotBasedLabel[VerbForm]] = {
    getSpanTans.get(span).toList.map { tan =>
      val subj = Option(question("subj")).filter(_ != "_").map(_.lowerCase)
      val isPassive = question("abst-verb") == "verb[pss]"
      val (wholeVerbPrefix, verbForm) = tan.getVerbPrefixAndForm(isPassive, subj.nonEmpty)
      SlotBasedLabel(
        wh = question("wh").lowerCase,
        aux = wholeVerbPrefix.headOption,
        subj = subj,
        verbPrefix = wholeVerbPrefix.drop(1),
        verb = verbForm,
        obj = Option(question("obj")).filter(_ != "_").map(_.lowerCase),
        prep = Option(question("prep")).filter(_ != "_").map(_.lowerCase),
        obj2 = Option(question("obj2")).filter(_ != "_").map(_.lowerCase))
    }
  }
}

object ClauseSlotMapper {
  def recapitalizeInflection(s: String): String = s match {
    // case "presentsingular3rd" => "presentSingular3rd"
    case "presentparticiple" => "presentParticiple"
    case "pastparticiple" => "pastParticiple"
    case x => x
  }
  // TODO: get rid of presentSingular3rd override once data is written correctly
  private[this] val verbStateMachine = new VerbTemplateStateMachine(
    InflectedForms.generic.copy(presentSingular3rd = "present".lowerCase), false, VerbTemplateStateMachine.TemplateComplete
  )
  private[this] val verbStringProcessor = new VerbStringProcessor(verbStateMachine)

  var cache = Map.empty[Map[String, String], Option[SlotBasedLabel[VerbForm]]]
  def apply(clauseSlots: Map[String, String]) = {
    cache.get(clauseSlots).getOrElse {
      val res = getQuestionSlots(clauseSlots)
      cache = cache + (clauseSlots -> res)
      res
    }
  }

  def getGenericFrame(question: Map[String, String]): Option[Frame] = {
    if(question.get("clause-" + question("clause-qarg")).exists(_ == "_")) None else {
      def getSlot(slot: String): Option[String] = Option(question(s"clause-$slot")).filter(_ != "_")
      def getNoun(slot: String) = getSlot(slot).map(s => Noun.fromPlaceholder(s.lowerCase).get)
      def getMisc = getSlot("misc").map(s => NonPrepArgument.fromPlaceholder(s.lowerCase).get)
      var argMap = DependentMap.empty[ArgumentSlot.Aux, Id]
        .put(Subj, getNoun("subj").get)
      getNoun("obj").foreach(n => argMap = argMap.put(Obj, n))
      getSlot("prep1").foreach { prep =>
        argMap = argMap.put(Prep1, Preposition(prep.lowerCase, getNoun("prep1-obj")))
      }
      getSlot("prep2").foreach { prep =>
        argMap = argMap.put(Prep2, Preposition(prep.lowerCase, getNoun("prep2-obj")))
      }
      getMisc.foreach { misc =>
        argMap = argMap.put(Misc, misc)
      }
      val verbString = List(getSlot("aux"), getSlot("verb")).flatten.mkString(" ")
      val verbResultOpt = verbStringProcessor.processStringFully(verbString).toOption
      if(verbResultOpt.isEmpty) {
        println(s"Could not resolve verb string ($verbString) for question: $question")
        None
      } else {
        val (isPassive, tan) = verbResultOpt
          .get.toList.collect {
            case VerbStringProcessor.CompleteState(_, _, isPassive, tan) =>
              (isPassive, tan)
          }.head
        Some(
          Frame(
            ArgStructure(argMap, isPassive),
            InflectedForms.generic, tan
          )
        )
      }
    }
  }

  def getQuestionSlots(
    question: Map[String, String]
  ): Option[SlotBasedLabel[VerbForm]] = {
    getGenericFrame(question).flatMap { frame =>
      val questionStrings = frame.questionsForSlot(ArgumentSlot.fromString(question("clause-qarg")).get)
      if(questionStrings.isEmpty) {
        println(frame, question("clause-qarg"))
      }
      val questionString = questionStrings.head
      val finalSlotsOpt = SlotBasedLabel.getVerbTenseAbstractedSlotsForQuestion(
        Vector(), InflectedForms.generic, List(questionString)
      ).head
      if(finalSlotsOpt.isEmpty) {
        println("Can't map question to slots: " + questionString)
      }
      finalSlotsOpt
    }
  }

  // def getSlot(slot: String): Option[String] = Option(question(s"clause-$slot")).filter(_ != "_")
  // def slotIsNot(slot: String) = question(s"clause-qarg") != slot
  // def filterSlot wh wh
  // val wh = if(question.keySet.contains(question("clause-qarg"))) {
  //   question("clause-" + question("clause-qarg"))
  // } else question("clause-qarg")
  // val (aux, verbPrefix, verb) = {
  //   if(slotIsNot("subj")) {
  //      (question("clause-aux"), ???, ???)
  //   } else {

  //   }
  // }
  // val prep = (optionalSlot("prep1"), optionalSlot("prep2")) match {
  //   case (None, None) => None
  //   case (Some(p), None) => Some(p)
  //   case (None, Some(p)) => Some(p)
  //   case (Some(p1), Some(p2)) =>
  //     if(optionalSlot("prep1-obj").nonEmpty && slotIsNot("prep1-obj")) {
  //       throw new RuntimeException("impossible to convert clause to QA-SRL question")
  //     }
  //     Some(s"$p1 $p2")
  // }
  // val obj2 = (optionalSlot("prep1-obj"), optionalSlot("prep2-obj"), optionalSlot("misc")) match {

  // }
  //   SlotBasedLabel(
  //     wh = wh.lowerCase
  //       aux = aux.lowerCase,
  //     subj = Option(question("clause-subj")).filter(_ => slotIsNot("subj")),
  //     verbPrefix = verbPrefix.map(_.lowerCase),
  //     verb = verbForm,
  //     obj = Option(question("clause-obj")).filter(_ => slotIsNot("obj")),
  //     prep = prep.map(_.lowerCase)
  //       obj2 = Option(question("obj2")).filter(_ != "_").map(_.lowerCase))
  // }
}

object QfirstClausalProtocol extends QfirstProtocol[Map[String, String]] {

  def getQuestions(
    question: Map[String, String],
    span: ESpan,
    beam: List[QfirstBeamItem[Map[String, String]]],
    getSpanTans: Option[ESpan => Set[TAN]],
    getAnimacy: Option[ESpan => Option[Boolean]],
    spanAnimacyThreshold: Double
  ): List[SlotBasedLabel[VerbForm]] = {
    ClauseSlotMapper(question).toList
  }
}

object QfirstClausalNoTanProtocol extends QfirstProtocol[Map[String, String]] {
  def isModal(t: TAN) = t.tense match {
    case Tense.Finite.Modal(_) => true
    case _ => false
  }
  def getQuestions(
    question: Map[String, String],
    span: ESpan,
    beam: List[QfirstBeamItem[Map[String, String]]],
    getSpanTans: Option[ESpan => Set[TAN]],
    getAnimacy: Option[ESpan => Option[Boolean]],
    spanAnimacyThreshold: Double
  ): List[SlotBasedLabel[VerbForm]] = {
    val isPassive = question("clause-abst-verb") == "verb[pss]"
    getSpanTans.get(span).toList
      .filter(tan => !isPassive || !tan.isProgressive || (!tan.isPerfect && !isModal(tan)))
      .flatMap { tan =>
      val (wholeVerbPrefix, verbForm) = tan.getVerbPrefixAndForm(isPassive, false)
      val verbString = ClauseSlotMapper.recapitalizeInflection(InflectedForms.generic(verbForm).toString)
      val aux = wholeVerbPrefix.headOption.fold("_")(_.toString)
      val verbSlot = NonEmptyList.fromList(wholeVerbPrefix.drop(1)) match {
        case None => verbString
        case Some(ws) => ws.toList.mkString(" ") + " " + verbString
      }
      var q = question
      q = q + ("clause-aux" -> aux)
      q = q + ("clause-verb" -> verbSlot)
      ClauseSlotMapper(q)
    }
  }
}

object QfirstClausalNoAnimProtocol extends QfirstProtocol[Map[String, String]] {

  def getQuestions(
    question: Map[String, String],
    span: ESpan,
    beam: List[QfirstBeamItem[Map[String, String]]],
    getSpanTans: Option[ESpan => Set[TAN]],
    getAnimacy: Option[ESpan => Option[Boolean]],
    spanAnimacyThreshold: Double
  ): List[SlotBasedLabel[VerbForm]] = {
    def getSlot(slot: String): Option[String] = Option(question(s"clause-$slot")).filter(_ != "_")
    val anim = getAnimacy.get
    val thisClause = question - "clause-qarg"
    val argToAnim = beam
      .filter(item => (item.questionSlots - "clause-qarg") == thisClause)
      .groupBy(item => item.questionSlots("clause-qarg"))
      .flatMap { case (answerSlot, items) =>
        val goodItems = items.filter(_.spanProb >= spanAnimacyThreshold)
        if(goodItems.nonEmpty) Some(answerSlot -> goodItems)
        else None
      }
      .flatMap { case (answerSlot, items) =>
        anim(items.maxBy(_.spanProb).span).map(answerSlot -> _)
      }
    var q = question
    def renderQ(q: Map[String, String]) = {
      List("abst-subj", "aux", "verb", "abst-obj", "prep1", "abst-prep1-obj", "prep2", "abst-prep2-obj", "abst-misc", "qarg")
        .map("clause-" + _).map(q).mkString(" ")
    }

    val subj = if(argToAnim.get("subj").getOrElse(true)) "someone" else "something"
    val obj = getSlot("abst-obj").fold("_")(_ => if(argToAnim.get("obj").getOrElse(false)) "someone" else "something")
    val prep1Obj = getSlot("abst-prep1-obj").fold("_")(_ => if(argToAnim.get("prep1-obj").getOrElse(false)) "someone" else "something")
    val prep2Obj = getSlot("abst-prep2-obj").fold("_")(_ => if(argToAnim.get("prep2-obj").getOrElse(false)) "someone" else "something")
    val misc = getSlot("abst-misc").fold("_")(s => if(s == "something" && argToAnim.get("misc").getOrElse(false)) "someone" else s)
    q = q + ("clause-subj" -> subj)
    q = q + ("clause-obj" -> obj)
    q = q + ("clause-prep1-obj" -> prep1Obj)
    q = q + ("clause-prep2-obj" -> prep2Obj)
    q = q + ("clause-misc" -> misc)
    ClauseSlotMapper(q).toList
  }
}

object QfirstClausalNoTanOrAnimProtocol extends QfirstProtocol[Map[String, String]] {
  def isModal(t: TAN) = t.tense match {
    case Tense.Finite.Modal(_) => true
    case _ => false
  }

  var numQuestionsPredicted = 0
  var numQuestionsEdited = 0
  var numDifferentQuestionsProduced = 0
  var numEditedQuestionsProduced = 0

  def getQuestions(
    question: Map[String, String],
    span: ESpan,
    beam: List[QfirstBeamItem[Map[String, String]]],
    getSpanTans: Option[ESpan => Set[TAN]],
    getAnimacy: Option[ESpan => Option[Boolean]],
    spanAnimacyThreshold: Double
  ): List[SlotBasedLabel[VerbForm]] = {
    def getSlot(slot: String): Option[String] = Option(question(s"clause-$slot")).filter(_ != "_")
    val spanTans = getSpanTans.get(span).toList
    val isPassive = if(question.contains("clause-verb")) {
      question("clause-verb").lowerCase.contains("pastParticiple".lowerCase) &&
        (Set("be", "been", "being").exists(question("clause-verb").contains) ||
           Set("is", "was").exists(question("clause-aux").contains))
    } else {
      question("clause-abst-verb") == "verb[pss]"
    }

    numQuestionsPredicted = numQuestionsPredicted + 1

    val tanResolvedQs = if(question.contains("clause-verb") && spanTans.isEmpty) {
      List(question)
    } else {
      val editedQs = spanTans
        .filter(tan => !isPassive || !tan.isProgressive || (!tan.isPerfect && !isModal(tan)))
        .map { tan =>
          val (wholeVerbPrefix, verbForm) = tan.getVerbPrefixAndForm(isPassive, false)
          val verbString = ClauseSlotMapper.recapitalizeInflection(InflectedForms.generic(verbForm).toString)
          val aux = wholeVerbPrefix.headOption.fold("_")(_.toString)
          val verbSlot = NonEmptyList.fromList(wholeVerbPrefix.drop(1)) match {
            case None => verbString
            case Some(ws) => ws.toList.mkString(" ") + " " + verbString
          }
          var q = question
          q = q + ("clause-aux" -> aux)
          q = q + ("clause-verb" -> verbSlot)
          q
        }

      if(editedQs.isEmpty) {
        List(question)
      } else {
        numQuestionsEdited = numQuestionsEdited + 1
        numEditedQuestionsProduced = numEditedQuestionsProduced + editedQs.size
        numDifferentQuestionsProduced = numDifferentQuestionsProduced + editedQs.filter(_ != question).size
        editedQs
      }
    }



    if(tanResolvedQs.isEmpty) Nil else {
      val anim = getAnimacy.get
      val thisClause = question - "clause-qarg"
      val argToAnim = beam
        .filter(item => (item.questionSlots - "clause-qarg") == thisClause)
        .groupBy(item => item.questionSlots("clause-qarg"))
        .flatMap { case (answerSlot, items) =>
          val goodItems = items.filter(_.spanProb >= spanAnimacyThreshold)
          if(goodItems.nonEmpty) Some(answerSlot -> goodItems)
          else None
        }
        .flatMap { case (answerSlot, items) =>
          anim(items.maxBy(_.spanProb).span).map(answerSlot -> _)
        }
      // argument animacy is not present in cases where there is no answer
      def getNounPlaceholder(isAnimate: Boolean) = if(isAnimate) "someone" else "something"
      val subj = argToAnim.get("subj").map(getNounPlaceholder).orElse(question.get("clause-subj")).getOrElse("someone")
      def getAnimacyRevisedNounArg(slot: String) = {
        question.get(s"clause-$slot").map { s =>
          // question has built-in animacy
          if(s == "_") "_" else argToAnim.get(slot).map(getNounPlaceholder).getOrElse(s)
        }.getOrElse {
          // question does not have built-in animacy
          getSlot(s"abst-$slot").fold("_")(_ => if(argToAnim.get(slot).getOrElse(false)) "someone" else "something")
        }
      }
      val obj = getAnimacyRevisedNounArg("obj")
      val prep1Obj = getAnimacyRevisedNounArg("prep1-obj")
      val prep2Obj = getAnimacyRevisedNounArg("prep2-obj")
      val misc = {
        question.get("clause-misc").map { s =>
          // question has built-in animacy
          if(Set("something", "someone").contains(s)) argToAnim.get("misc").map(getNounPlaceholder).getOrElse(s)
          else s
        }.getOrElse {
          getSlot("abst-misc").fold("_")(s => if(s == "something" && argToAnim.get("misc").getOrElse(false)) "someone" else s)
        }
      }

      tanResolvedQs.flatMap { tanQ =>
        var q = tanQ
        q = q + ("clause-subj" -> subj)
        q = q + ("clause-obj" -> obj)
        q = q + ("clause-prep1-obj" -> prep1Obj)
        q = q + ("clause-prep2-obj" -> prep2Obj)
        q = q + ("clause-misc" -> misc)
        ClauseSlotMapper(q)
      }
    }
  }
}
