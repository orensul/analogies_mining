package qfirst.clause.ext.demo
import qfirst.clause.ext._

import qasrl.crowd.util._

import qasrl.bank.SentenceId

import japgolly.scalajs.react.vdom.html_<^._
import japgolly.scalajs.react._

import scalacss.DevDefaults._
import scalacss.ScalaCssReact._

object Dispatcher extends ClausalDispatcher[SentenceId] {

  val dataToggle = VdomAttr("data-toggle")
  val dataPlacement = VdomAttr("data-placement")

  val TooltipsComponent = ScalaComponent
    .builder[VdomTag]("Tooltips")
    .render(_.props)
    .componentDidMount(
      _ =>
        Callback {
          scala.util.Try {
            scala.scalajs.js.Dynamic.global.$("[data-toggle=\"tooltip\"]").tooltip()
          }
          ()
      }
    )
    .build

  import settings._

  def example(question: String, answer: String, isGood: Boolean, tooltip: String = "") =
    <.li(
      <.span(
        if (isGood) Styles.goodGreen else Styles.badRed,
        TagMod(
          Styles.underlined,
          dataToggle := "tooltip",
          dataPlacement := "top",
          ^.title := tooltip
        ).when(tooltip.nonEmpty),
        <.span(question),
        <.span(" --> "),
        <.span(answer)
      )
    )

  private[this] val examples = <.div(
    TooltipsComponent(
      <.div(
        <.p(
          Styles.bolded,
          " This section is exactly the same between the question writing and question answering tasks. "
        ),
        <.p(
          " Below, for each verb, we list a complete set of good questions (green) and some bad ones (red). ",
          " Hover the mouse over the underlined examples for an explanation. "
        ),
        <.blockquote(
          ^.classSet1("blockquote"),
          "Protesters ",
          <.span(Styles.bolded, " blamed "),
          " the corruption scandal on local officials, who today refused to promise that they would resume the investigation before year's end. "
        ),
        <.ul(
          example("Who blamed someone?", "Protesters", true),
          example("Who did someone blame something on?", "local officials", true),
          example(
            "What did someone blame someone for?",
            "the corruption scandal",
            true,
            """ "What did someone blame on someone?" would also have been okay. """
          ),
          example(
            "Who blamed?",
            "Protesters",
            false,
            """ This question is invalid by the litmus test, because the sentence "Protesters blamed." is ungrammatical. """
          )
        ),
        <.blockquote(
          ^.classSet1("blockquote"),
          "Protesters blamed the corruption scandal on local officials, who today ",
          <.span(Styles.bolded, " refused "),
          " to promise that they would resume the investigation before year's end. "
        ),
        <.ul(
          example(
            "Who refused to do something?",
            "local officials / they",
            true,
            """When answering, list all of the phrases in the sentence that refer to the correct answer, including pronouns like "they"."""
          ),
          example(
            "What did someone refuse to do?",
            "promise that they would resume the investigation before year's end",
            true
          ),
          example(
            "What did someone refuse to do?",
            "promise that they would resume the investigation",
            false,
            """The answer is not specific enough: it should include "before year's end" because that was part of what they were refusing to promise."""
          ),
          example(
            "What did someone refuse to do?",
            "resume the investigation before year's end",
            false,
            """This answer is also bad: you should instead choose the more literal answer above."""
          ),
          example("When did someone refuse to do something?", "today", true),
          example(
            "Who didn't refuse to do something?",
            "Protesters",
            false,
            """The sentence does not say anything about protesters refusing or not refusing, so this question is invalid."""
          )
        ),
        <.blockquote(
          ^.classSet1("blockquote"),
          "Protesters blamed the corruption scandal on local officials, who today refused to ",
          <.span(Styles.bolded, " promise "),
          " that they would resume the investigation before year's end. "
        ),
        <.ul(
          example(
            "Who didn't promise something?",
            "local officials / they",
            true,
            "Negated questions work when the sentence is indicating that the event or state expressed by the verb did not happen."
          ),
          example(
            "What didn't someone promise?",
            "that they would resume the investigation before year's end",
            true
          ),
          example(
            "When didn't someone promise to do something?",
            "before year's end",
            false,
            """ This question is bad because "before year's end" refers to the timeframe of resuming the investigation, not the timeframe of the promise being made.
            All such questions must pertain to the time/place of the chosen verb. """
          )
        ),
        <.blockquote(
          ^.classSet1("blockquote"),
          "Protesters blamed the corruption scandal on local officials, who today refused to promise that they would ",
          <.span(Styles.bolded, " resume "),
          " the investigation before year's end. "
        ),
        <.ul(
          example(
            "Who might resume something?",
            "local officials / they",
            true,
            """Words like "might" or "would" are appropriate when the sentence doesn't clearly indicate whether something actually happened."""
          ),
          example("What might someone resume?", "the investigation", true),
          example("When might someone resume something?", "before year's end", true)
        ),
        <.blockquote(
          ^.classSet1("blockquote"),
          <.span(Styles.bolded, " Let"),
          "'s go up to the counter and ask."
        ),
        <.ul(
          example(
            "Who should someone let do something?",
            "'s",
            true,
            """Here, you should read 's as the word it stands for: "us".
            So by substituting back into the question, we get "someone should let us do something",
            which is what someone is suggesting when they say "Let's go". """
          ),
          example(
            "What should someone let someone do?",
            "go up to the counter and ask",
            true,
            """It would also be acceptable to mark "go up to the counter" and "ask" as two different answers. """
          ),
          example(
            "Where should someone let someone do something?",
            "the counter",
            false,
            """Questions should only concern the targeted verb: "letting" is not happening at the counter."""
          )
        ),
        <.blockquote(
          ^.classSet1("blockquote"),
          "Let's ",
          <.span(Styles.bolded, " go "),
          " up to the counter and ask."
        ),
        <.ul(
          example("Who should go somewhere?", "'s", true),
          example(
            "Where should someone go?",
            "up to the counter",
            true,
            """Since both "up" and "to the counter" describe where they will go, they should both be included in the answer to a "where" question. """
          )
        )
      )
    )
  )

  val validationOverview = <.div(
    <.p(
      Styles.badRed,
      """Read through all of the instructions and make sure you understand the interface controls before beginning. A full understanding of the requirements will help maximize your agreement with other workers so you can retain your qualification."""
    ),
    <.p(
      s"""This task is for an academic research project by the natural language processing group at the University of Washington.
           We wish to deconstruct the meanings of English sentences into lists of questions and answers.
           You will be presented with a selection of English text and a list of questions prepared by other annotators."""
    ),
    <.p(
      """You will highlight the words in the sentence that correctly answer each question,
           as well as mark whether questions are invalid.""",
      <.b(
        """ Note: it takes exactly 2 clicks to highlight each answer; see the Controls tab for details. """
      ),
      """For example, consider the following sentence:"""
    ),
    <.blockquote(
      ^.classSet1("blockquote"),
      "Protesters ",
      <.span(Styles.bolded, " blamed "),
      " the corruption scandal on local officials, who today ",
      " refused to promise that they would resume the investigation before year's end. "
    ),
    <.p("""You should choose all of the following answers:"""),
    <.ul(
      <.li("Who blamed someone? --> ", <.span(Styles.goodGreen, " Protesters ")),
      <.li(
        "Who did someone blame something on? --> ",
        <.span(Styles.goodGreen, " local officials / they")
      ),
      <.li(
        "What did someone blame on someone? --> ",
        <.span(Styles.goodGreen, " the corruption scandal")
      )
    ),
    <.p(
      s"""You will be paid a ${dollarsToCents(validationBonusPerQuestion)}c bonus per question after the first $validationBonusThreshold questions if there are more than $validationBonusThreshold."""
    ),
    <.h2("""Guidelines"""),
    <.ol(
      <.li(
        <.span(Styles.bolded, "Correctness. "),
        """Each answer must satisfy the litmus test that if you substitute it back into the question,
           the result is a grammatical statement, and it is true according to the sentence given. For example, """,
        <.span(Styles.bolded, "Who blamed someone? --> Protesters"),
        """ becomes """,
        <.span(Styles.goodGreen, "Protesters blamed someone, "),
        """ which is valid, while """,
        <.span(Styles.bolded, "Who blamed? --> Protesters"),
        """ would become """,
        <.span(Styles.badRed, "Protesters blamed, "),
        s""" which is ungrammatical, so it is invalid.
           Your responses will be compared to other annotators, and you must agree with them
           ${(100.0 * validationAgreementBlockingThreshold).toInt}% of the time in order to remain qualified. """
      ),
      <.li(
        <.span(Styles.bolded, "Verb-relevance. "),
        """ Answers to the questions must pertain to the participants, time, place, reason, etc., of """,
        <.span(Styles.bolded, " the target verb in the sentence, "),
        " which is bolded and colored blue in the interface. ",
        """ For example, if the sentence is """,
        <.span(
          Styles.bolded,
          " He ",
          <.span(Styles.niceBlue, Styles.underlined, "promised"),
          " to come tomorrow "
        ),
        """ and the question is """,
        <.span(Styles.badRed, " When did someone promise to do something? "),
        """ you must mark it """,
        <.span(Styles.badRed, " Invalid "),
        """ because the time mentioned, """,
        <.i(" tomorrow, "),
        " is ",
        <.i(" not "),
        " the time that he made the promise, but rather the time that he might come."
      ),
      <.li(
        <.span(Styles.bolded, "Exhaustiveness. "),
        s"""You must provide every possible answer to each question.
           When highlighting answers, please only include the necessary words to provide a complete, grammatical answer,
           but if all else is equal, prefer to use longer answers.
           Also please include pronouns in the sentence that refer an answer you've already given.
           However, note that none of the answers to your questions may overlap.
           If the only possible answers to a question were already used for previous questions, please mark it invalid."""
      )
    ),
    <.p(
      " All ungrammatical questions should be counted invalid. However, ",
      " If the sentence has grammatical errors or is not a complete sentence, please answer ",
      " questions according to the sentence's meaning to the best of your ability. "
    ),
    <.p("Please read through the examples if you need more details.")
  )

  val validationControls = <.div(
    <.ul(
      <.li(
        <.span(Styles.bolded, "Navigation. "),
        "Change questions using the mouse, the up and down arrow keys, or W and S."
      ),
      <.li(
        <.span(Styles.bolded, "Invalid Questions. "),
        "Click the button labeled \"Invalid\" or press the space bar to toggle a question as invalid."
      ),
      <.li(
        <.span(Styles.bolded, "Answers. "),
        "To highlight an answer, first click on the first word in the answer, which will turn ",
        <.span(^.backgroundColor := "#FF8000", "orange"),
        ". Then click on the last word in the answer (which may be the same word) and the whole phrase will turn ",
        <.span(^.backgroundColor := "#FFFF00", "yellow"),
        ". (You may also click them in the opposite order.) You can highlight multiple answers to the same question in this way. ",
        " To delete an answer, click on a word in that answer while it is highlighted yellow. ",
        """ None of your answers may overlap with each other; answers to questions other than the currently selected one
        will be highlighted in """,
        <.span(^.backgroundColor := "#DDDDDD", "grey"),
        "."
      )
    )
  )

  val validationConditions = <.div(
    <.p(s"""You will be paid a bonus of ${dollarsToCents(validationBonusPerQuestion)}c
        for every question beyond $validationBonusThreshold, which will be paid when the assignment is approved.
        Your judgments will be cross-checked with other workers,
        and your agreement rate will be shown to you in the interface.
        If this number drops below ${(100 * validationAgreementBlockingThreshold).toInt}
        you will no longer qualify for the task.
        (Note that other validators will sometimes make mistakes,
        so there is an element of randomness to it: don't read too deeply into small changes in your agreement rate.)
        Your work will be approved and the bonus will be paid within an hour.""")
  )

  override val evaluationInstructions = <.div(^.visibility := "hidden")
  // override val evaluationInstructions = <.div(
  //   InstructionsPanel.Component(
  //     InstructionsPanel.Props(
  //       instructionsId = "instructions",
  //       collapseCookieId = "validationCollapseCookie",
  //       tabs = List(
  //         "Overview"             -> validationOverview,
  //         "Controls"             -> validationControls,
  //         "Conditions & Payment" -> validationConditions,
  //         "Examples"             -> examples
  //       )
  //     )
  //   )
  // )
}
