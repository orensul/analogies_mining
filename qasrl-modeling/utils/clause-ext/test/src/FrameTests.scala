package qfirst.clause.ext

import jjm.LowerCaseString
import jjm.ling.en.InflectedForms
import jjm.implicits._

import cats.implicits._

import scala.util.{Success, Try}

import java.nio.file.Paths

import org.scalatest._
import org.scalatest.prop._

class FrameTests extends FunSuite with Matchers {

  import org.scalatest.Inside._
  import org.scalatest.AppendedClues._

  val questionsTry = Try(scala.io.Source.fromURL(getClass.getResource("/question-strings.txt")).getLines.toList)
  val badQuestionsTry = Try(scala.io.Source.fromURL(getClass.getResource("/bad-question-strings.txt")).getLines.toList)

  test("reference questions can be read from resources") {
    inside(questionsTry) {
      case Success(lines) => lines should have size 13611
    }
  }

  // TODO test bad questions invalid

  val questions = questionsTry.get

  def processQuestion(question: String) = {
    val questionTokensIsh = question.init.split(" ").toVector
    val stateMachine = new TemplateStateMachine(questionTokensIsh, InflectedForms.generic)
    val template = new QuestionProcessor(stateMachine)

    template.processStringFully(question)
  }

  val processedQuestionResults = questions.map(processQuestion)

  val (invalidQuestionsWithStates, goodQuestionsWithStates) = (questions.zip(processedQuestionResults)).map {
    case (q, Left(x)) => Left(q -> x)
    case (q, Right(x)) => Right(q -> x)
  }.separate
  val (invalidQuestions, invalidStates) = invalidQuestionsWithStates.unzip
  val (goodQuestions, goodStates) = goodQuestionsWithStates.unzip

  val splitGoodStates = goodStates.map(
    _.map(QuestionProcessor.ValidState.eitherIso.get).toList.separate
  )

  test("all reference questions are valid") {
    val x = invalidQuestions.isEmpty
    val questionsWord = if(invalidStates.size == 1) "question" else "questions"
    assert(x) withClue (
      s"for ${invalidQuestions.size} $questionsWord:\n" +
        invalidQuestions.take(10).map("  " + _).mkString("\n")
    )
  }

  test("no reference questions are incomplete") {
    splitGoodStates.foreach { case (inProgress, _) =>
      inProgress should have size 0
    }
  }

  val completeStates = splitGoodStates.map(_._2)

  test("all reference questions have some complete state") {
    completeStates.foreach(_ should not be empty)
  }

  test("frames are not repeated in processor output") {
    completeStates.foreach(states =>
      states.size == states.toSet.size
    )
  }

  test("all complete states should reproduce the reference question text") {
    goodQuestions.zip(completeStates).foreach { case (q, state) =>
      state.foreach(_.fullText shouldEqual q)
    }
  }

  // old stuff for comparison
  // val completeStatesOld = {
  //   def processQuestion(question: String) = {
  //     val questionTokensIsh = question.init.split(" ").toVector
  //     val stateMachine = new qasrl.TemplateStateMachine(questionTokensIsh, InflectedForms.generic)
  //     val template = new qasrl.QuestionProcessor(stateMachine)

  //     template.processStringFully(question)
  //   }
  //   val processedQuestionResults = questions.map(processQuestion)

  //   val (invalidQuestionsWithStates, goodQuestionsWithStates) = (questions.zip(processedQuestionResults)).map {
  //     case (q, Left(x)) => Left(q -> x)
  //     case (q, Right(x)) => Right(q -> x)
  //   }.separate
  //   val (invalidQuestions, invalidStates) = invalidQuestionsWithStates.unzip
  //   val (goodQuestions, goodStates) = goodQuestionsWithStates.unzip

  //   val splitGoodStates = goodStates.map(
  //     _.map(qasrl.QuestionProcessor.ValidState.eitherIso.get).toList.separate
  //   )
  //   val completeStates = splitGoodStates.map(_._2)
  //   completeStates
  // }

  // var x = 0
  // for((oldState, state) <- completeStatesOld.zip(completeStates)) {
  //   val oldStateSet = oldState.toSet
  //   if(oldStateSet.size != state.size) {
  //     println
  //     println(state.head.fullText)
  //     println("Old:")
  //     oldStateSet.foreach { s =>
  //       println("\t" + s.frame.args.toString + ": " + s.answerSlot)
  //     }
  //     println("New:")
  //     state.foreach { s =>
  //       println("\t" + s.frame.args.toString + ": " + s.answerSlot)
  //     }
  //     x += 1
  //   }
  // }
  // println(s"$x questions differ in num states")

  // completeStates.filter(_.size > 3).foreach { state =>
  //   println(state.head.fullText)
  //   state.foreach { s =>
  //     println("\t" + s.frame.args.toString + ": " + s.answerSlot)
  //   }
  // }

  // val completeStatesReferenceHistOld = Map(
  //   1 -> 10882,
  //   2 -> 2599,
  //   3 -> 135
  // )

  val completeStatesReferenceHist = Map(
    1 -> 10765,
    2 -> 2478,
    3 -> 367,
    5 -> 1
  )
  val completeStatesHist = completeStates.map(_.size).groupBy(x => x).map { case (k, v) => k -> v.size }

  val uniqueFrameArgPairs = completeStates.flatten.toSet
  val uniqueFrames = uniqueFrameArgPairs.map(_.frame)
  val uniqueFrameTemplates = uniqueFrames.map(_.structure)
  val uniqueFrameTemplateArgPairs = uniqueFrameArgPairs.map(s => s.frame.structure -> s.answerSlot).size

  println(s"Unique questions: " + questions.size)
  println(s"Unique frame-arg pairs: " + uniqueFrameArgPairs.size)
  println(s"Unique frames: " + uniqueFrames.size)
  println(s"Unique frame template - arg pairs: " + uniqueFrameTemplateArgPairs)
  println(s"Unique frame templates: " + uniqueFrameTemplates.size)

  test("complete states have the expected number of frames") {
    completeStatesHist shouldEqual completeStatesReferenceHist
  }

  import QuestionProcessor.CompleteState

  def getSatisfyingStateLists(p: CompleteState => Boolean) = {
    completeStates.map(_.filter(p)).filter(_.nonEmpty)
  }

  def getStateSetClue(s: List[List[CompleteState]]) = {
    val numQs = s.size
    val questionWord = if(numQs == 1) "question" else "questions"
    val numQsToPrint = 10
    s"for $numQs $questionWord:\n" + (
      s.map(set =>
        f"${set.head.fullText}%-60s " + set.toList.map(s =>
          // after first tab if rendering doesn't work
          // + s.frame.questionsForSlot(s.answerSlot) + "\t"
          "\n\t" + s.frame.args.toString + ": " + s.answerSlot
        ).mkString
      ).map("  " + _).take(numQsToPrint).mkString("\n") + (
        if(s.size > numQsToPrint) "\n  ...\n" else "\n"
      )
    )
  }


  def assertAllStatesSatisfy(p: CompleteState => Boolean) = {
    val satisfyingStateLists = getSatisfyingStateLists(x => !p(x))
    val x = satisfyingStateLists.isEmpty
    assert(x) withClue getStateSetClue(satisfyingStateLists)
  }

  test("re-rendering question from induced frame yields original question") {
    assertAllStatesSatisfy(s =>
      s.frame.questionsForSlot(s.answerSlot).forall(_ == s.fullText)
    )
  }

  test("prep2 can only appear in presence of prep1") {
    assertAllStatesSatisfy(s =>
      s.frame.args.get(Prep2).isEmpty || s.frame.args.get(Prep1).nonEmpty
    )
  }

  test("obj2 can only appear if there is an obj1 and there is no prep") {
    assertAllStatesSatisfy(s =>
      !s.frame.args.get(Misc).exists(_.isNoun) || (
        s.frame.args.get(Obj).nonEmpty && s.frame.args.get(Prep1).isEmpty
      )
    )
  }

  {
    import qasrl.bank.Data
    import qasrl.data.Dataset
    val train = Data.readQasrlDataset(Paths.get("../qasrl-bank/data/qasrl-v2_1").resolve("orig").resolve("dev.jsonl.gz")).get
    Dataset.verbEntries.getAll(train).foreach { verb =>
      verb.questionLabels.keys.foreach { question =>
        val questionTokensIsh = question.init.split(" ").toVector
        val stateMachine = new TemplateStateMachine(questionTokensIsh, verb.verbInflectedForms)
        val template = new QuestionProcessor(stateMachine)
        val results = template.processStringFully(question)
        val isValid = results.isRight
        if(!isValid) println(question)
      }
    }
  }


  // test("no frames have whitespace prepositions") {
  //   assertNoSatisfyingStates(s =>
  //     s.frame.args.get(Prep1).collect {
  //       case Prep(p, _) => p.trim == ""
  //     }.exists(identity)
  //   )
  //   assertNoSatisfyingStates(s =>
  //     s.frame.args.get(Prep2).collect {
  //       case Prep(p, _) => p.trim == ""
  //     }.exists(identity)
  //   )
  // }

  // // NOTE: actually right now we allow this
  // // old: false for 1689 questions
  // test("no frames have do/doing preps") {
  //   assertNoSatisfyingStates(s =>
  //     s.frame.args.get(Obj2).collect {
  //       case Prep(p, _) => p.toString.endsWith("do") || p.toString.endsWith("doing")
  //     }.exists(identity)
  //   )
  // }

  // val slotLists = completeStates.map(_.map(s => SlotBasedLabel.getSlotsForQuestionStructure(s.frame, s.answerSlot)))

  // test("slots are deterministic for a question") {
  //   val nonDeterministicSlotLists = slotLists.map(_.toSet)
  //     .filter(_.size != 1) // 95 left
  //     .filter { slotSet =>
  //     slotSet.toList match {
  //       // remove the case where obj and obj2 are switched with an empty prep in between
  //       case fst :: snd :: Nil =>
  //         !(
  //           fst.prep.isEmpty && snd.prep.isEmpty && fst.obj == snd.obj2 && snd.obj == fst.obj2 && (
  //             fst.copy(obj = snd.obj, obj2 = snd.obj2) == snd
  //           )
  //         )
  //       case _ => true
  //     }
  //   }
  //   val questionsWord = if(nonDeterministicSlotLists.size == 1) "question" else "questions"
  //   val x = nonDeterministicSlotLists.isEmpty
  //   assert(x) withClue (
  //     s"for ${nonDeterministicSlotLists.size} $questionsWord:\n" +
  //       nonDeterministicSlotLists.take(10).map(_.map(_.renderWithSeparator(identity, ",")).mkString("\t")).map("  " + _).mkString("\n")
  //   )
  // }

  // val prepHasWhitespace = (s: SlotBasedLabel[LowerCaseString]) => s.prep.exists(p => p.toString != p.trim)
  // val prepIsWhitespace = (s: SlotBasedLabel[LowerCaseString]) => s.prep.exists(_.trim == "")
  // val prepContainsDo =   (s: SlotBasedLabel[LowerCaseString]) => {
  //   s.prep.exists(_.toString.endsWith("do")) || s.prep.exists(_.toString.endsWith("doing"))
  // }
  // val toDoIsSplit = (s: SlotBasedLabel[LowerCaseString]) => {
  //   s.prep.map(_.toString).exists(p => p.endsWith(" to") || p == "to") &&
  //     s.obj2.map(_.toString).exists(o => o.startsWith("do ") || o == "do")
  // }
  // val hasDoSomeone = (s: SlotBasedLabel[LowerCaseString]) => {
  //   s.obj2.exists(o => o == "do someone".lowerCase || o == "doing someone".lowerCase)
  // }

  // val slotsPassMuster =  (s: SlotBasedLabel[LowerCaseString]) =>
  //   !(prepHasWhitespace(s) || prepIsWhitespace(s) || prepContainsDo(s) || toDoIsSplit(s))

  // def getSatisfyingSlotLists(p: SlotBasedLabel[LowerCaseString] => Boolean) = {
  //   slotLists.map(_.filter(p)).filter(_.nonEmpty)
  // }

  // def getSlotSetClue(s: List[List[SlotBasedLabel[LowerCaseString]]]) = {
  //   val numQs = s.foldMap(_.size)
  //   val questionWord = if(numQs == 1) "question" else "questions"
  //   val numQsToPrint = 10
  //   s"for $numQs $questionWord:\n" + (
  //     s.map(set =>
  //       set.toList.map(_.renderWithSeparator(identity, ",")).mkString("\t|\t")
  //     ).map("  " + _).take(numQsToPrint).mkString("\n") + (
  //       if(s.size > numQsToPrint) "\n  ...\n" else "\n"
  //     )
  //   )
  // }

  // def assertNoSatisfyingSlots(p: SlotBasedLabel[LowerCaseString] => Boolean) = {
  //   val satisfyingSlotLists = getSatisfyingSlotLists(p)
  //   val x = satisfyingSlotLists.isEmpty
  //   assert(x) withClue getSlotSetClue(satisfyingSlotLists)
  // }

  // test("slots reconstruct original question") {
  //   assertNoSatisfyingStates(s => s.fullText != SlotBasedLabel.getSlotsForQuestionStructure(s.frame, s.answerSlot).renderQuestionString(identity))
  // }

  // // old: failed for 136 questions
  // test("no prepositions are whitespace/empty") {
  //   assertNoSatisfyingSlots(prepIsWhitespace)
  // }

  // test("no prepositions have extra whitespace") {
  //   assertNoSatisfyingSlots(prepHasWhitespace)
  // }

  // test("no prepositions contain do/doing") {
  //   assertNoSatisfyingSlots(prepContainsDo)
  // }

  // test("\"to do\" is not split between slots") {
  //   assertNoSatisfyingSlots(toDoIsSplit)
  // }

  // test("\"do/doing someone\" never appears") {
  //   assertNoSatisfyingSlots(hasDoSomeone)
  // }
}
