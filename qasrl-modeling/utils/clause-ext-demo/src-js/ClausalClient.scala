package qfirst.clause.ext.demo
import qfirst.clause.ext._

import qfirst.model.eval.protocols.ClauseSlotMapper

import qasrl.crowd._
import qasrl.crowd.util.MultiContigSpanHighlightableSentenceComponent
import qasrl.crowd.util.Styles
import qasrl.crowd.util.implicits._

import spacro.tasks._

import cats.implicits._

import jjm.OrWrapped
import jjm.ling.{ESpan, ISpan}
import jjm.ling.Text
import jjm.ui._
import jjm.implicits._

import scalajs.js
import org.scalajs.dom
import org.scalajs.dom.raw._
import org.scalajs.dom.ext.KeyCode
import org.scalajs.jquery.jQuery

import scala.concurrent.ExecutionContext.Implicits.global

import japgolly.scalajs.react.vdom.html_<^._
import japgolly.scalajs.react._

import scalacss.DevDefaults._
import scalacss.ScalaCssReact._

import monocle._
import monocle.function.{all => Optics}
import monocle.macros._
import japgolly.scalajs.react.extra.StateSnapshot
import japgolly.scalajs.react.MonocleReact._
import japgolly.scalajs.react.CatsReact._

import io.circe.{Encoder, Decoder}

class ClausalClient[SID : Encoder : Decoder](instructions: VdomTag)(
  implicit settings: QASRLEvaluationSettings,
  promptDecoder: Decoder[ClausalPrompt[SID]], // macro serializers don't work for superclass constructor parameters
  responseEncoder: Encoder[List[QASRLValidationAnswer]], // same as above
  ajaxRequestEncoder: Encoder[ClausalAjaxRequest[SID]] // "
) extends TaskClient[ClausalPrompt[SID], List[QASRLValidationAnswer], ClausalAjaxRequest[SID]] {

  def main(): Unit = jQuery { () =>
    Styles.addToDocument()
    FullUI().renderIntoDOM(dom.document.getElementById(FieldLabels.rootClientDivLabel))
  }

  case class Question(
    verbIndex: Int,
    string: String
  )

  @Lenses case class State(
    curFocus: (Int, Int),
    isInterfaceFocused: Boolean,
    answers: List[List[QASRLValidationAnswer]]
  )

  object State {
    def initial(numQs: List[Int]) = State((0, 0), false, numQs.map(n => List.fill(n)(Answer(List.empty[ISpan]))))
  }

  val StateLocal = new LocalState[State]
  val DoubleLocal = new LocalState[Double]
  val StringLocal = new LocalState[String]
  val BoolLocal = new LocalState[Boolean]
  val AjaxComponent = new CacheCallContent[ClausalAjaxRequest[SID], ClausalAjaxResponse]
  val SpanHighlighting = new SpanSelection[(Int, Int)] // clause group, question
  import MultiContigSpanHighlightableSentenceComponent._

  def checkboxToggle(
    label: String,
    isValueActive: StateSnapshot[Boolean]
  ) = <.span(
    <.input(
      ^.`type` := "checkbox",
      ^.value := label,
      ^.checked := isValueActive.value,
      ^.onChange --> isValueActive.modState(!_)
    ),
    <.span(
      label
    )
  )

  def liveTextField[A](
    label: Option[String],
    value: StateSnapshot[A],
    makeValue: String => Option[A]
  ) = {
    <.span(
      label.whenDefined, " ",
      StringLocal.make(initialValue = value.value.toString) { inputText =>
        BoolLocal.make(initialValue = false) { isInvalid =>
          <.input(/* Styles.textField, */Styles.badRed.when(isInvalid.value))(
            ^.`type` := "text",
            ^.value := inputText.value,
            ^.onChange ==> ((e: ReactEventFromInput) =>
              inputText.setState(e.target.value) >>
                makeValue(e.target.value).fold(isInvalid.setState(true))(v =>
                  isInvalid.setState(false) >> value.setState(v)
                )
            )
          )
        }
      }
    )
  }

  def validationAnswerOptics(focus: (Int, Int)) = State.answers
    .composeOptional(Optics.index(focus._1))
    .composeOptional(Optics.index(focus._2))

  def answerSpanOptics(focus: (Int, Int)) = validationAnswerOptics(focus)
    .composePrism(QASRLValidationAnswer.answer)
    .composeLens(Answer.spans)

  def updateResponse(state: State): Callback = Callback(setResponse(state.answers.flatten))

  def updateCurrentAnswers(
    state: StateSnapshot[State])(
    highlightingState: SpanHighlighting.State
  ) = state.modState(
    answerSpanOptics(state.value.curFocus).set(
      highlightingState.spans(state.value.curFocus)
    )
  )

  def toggleInvalidAtFocus(
    state: StateSnapshot[State]) (
    highlightedAnswers: Map[(Int, Int), Answer])(
    focus: (Int, Int)
  ) = state.modState(
    validationAnswerOptics(focus).modify(va =>
      if (va.isInvalid) highlightedAnswers(focus)
      else InvalidQuestion
    )
  )

  def qaField(
    s: StateSnapshot[State], sentence: Vector[String], verbIndex: Int, question: String, prob: Double, highlightedAnswers: Map[(Int, Int), Answer])(
    index: (Int, Int)
  ) = {
    val isFocused = s.value.curFocus == index
    val answer = validationAnswerOptics(index).getOption(s.value).get

    <.div(
      ^.overflow := "hidden",
      <.span(
        Styles.bolded.when(isFocused),
        Styles.unselectable,
        ^.float := "left",
        ^.margin := "1px",
        ^.padding := "1px",
        ^.onClick --> s.modState(State.curFocus.set(index)),
        f"($prob%.2f) $question%s"
      )
    )
  }

  case class ClauseGroup(
    frame: Frame,
    verbIndex: Int,
    slotProbs: Map[ArgumentSlot, Double],
    slotInvalidProbs: Map[ArgumentSlot, Double],
    slotSpans: Map[ArgumentSlot, List[(ESpan, Double)]]
  ) {
    def aggregateProb = slotProbs.map(_._2).sum
    def bestSpansAboveThreshold(thresh: Double) = slotSpans.flatMap {
      case (slot, spans) => spans.filter(_._2 >= thresh)
          .sortBy(-_._2).headOption.map(_._1)
          .map(slot -> _)
    }.toMap
  }

  val body = {
    AjaxComponent.make(ClausalAjaxRequest(workerIdOpt, prompt.sentenceId), r => OrWrapped.wrapped(AsyncCallback.fromFuture(makeAjaxRequest(r)))) {
      case AjaxComponent.Loading => <.div("Retrieving data...")
      case AjaxComponent.Loaded(ClausalAjaxResponse(workerInfoSummaryOpt, sentence)) =>
        val clauseGroups = sentence.verbs.toList.sortBy(_.verbIndex).flatMap { verb =>
          verb.beam.qa_beam
            .groupBy(pred => ClauseSlotMapper.getGenericFrame(pred.questionSlots)).toList
            .collect { case (Some(frame), preds) => frame -> preds.groupBy(p =>
                ArgumentSlot.fromString(p.questionSlots("clause-qarg")).get
              )
            }
            .sortBy(-_._2.toList.map(_._2.head.questionProb).sum)
            .map { case (frame, predsByQArg) =>
              ClauseGroup(
                frame.copy(verbInflectedForms = verb.verbInflectedForms),
                verb.verbIndex,
                predsByQArg.transform { case (qarg, preds) => preds.head.questionProb },
                predsByQArg.transform { case (qarg, preds) => preds.head.invalidProb },
                predsByQArg.transform { case (qarg, preds) => preds.map(p => p.span -> p.spanProb) }
              )
            }
        }

        def getRemainingInAgreementGracePeriodOpt(summary: QASRLValidationWorkerInfoSummary) =
          Option(settings.validationAgreementGracePeriod - summary.numAssignmentsCompleted)
            .filter(_ > 0)

        DoubleLocal.make(0.1) { clauseInclusionThreshold =>
          DoubleLocal.make(0.0) { questionInclusionThreshold =>
            DoubleLocal.make(0.5) { spanInclusionThreshold =>
              BoolLocal.make(true) { showDiscourseQuestions =>
                StateLocal.make(State.initial(clauseGroups.map(_.slotProbs.size))) { state =>
                  import state.value._
                  SpanHighlighting.make(
                    isEnabled = !isNotAssigned && answers(curFocus._1)(curFocus._2).isAnswer,
                    enableSpanOverlap = true,
                    update = updateCurrentAnswers(state)) {
                    case (hs @ SpanHighlighting.State(spans, status), SpanHighlighting.Context(_, hover, touch, cancelHighlight)) =>
                      val curVerbIndex = clauseGroups(curFocus._1).verbIndex
                      val inProgressAnswerOpt =
                        SpanHighlighting.Status.selecting.getOption(status).map {
                          case SpanHighlighting.Selecting(_, anchor, endpoint) => ISpan(anchor, endpoint)
                        }
                      val curAnswers = spans(curFocus)
                      val otherAnswers = (spans - curFocus).values.flatten
                      val highlightedAnswers = clauseGroups.zipWithIndex.flatMap {
                        case (cg, cIndex) => cg.slotProbs.toList.indices.map { qIndex =>
                          (cIndex, qIndex) -> Answer(spans(cIndex -> qIndex))
                        }
                      }.toMap

                      val isCurrentInvalid = answers(curFocus._1)(curFocus._2).isInvalid
                      val touchWord = touch(curFocus)

                      <.div(
                        ^.classSet1("container-fluid"),
                        ^.onClick --> cancelHighlight,
                        <.div(
                          ^.classSet1("card"),
                          ^.margin := "5px",
                          ^.padding := "5px",
                          ^.tabIndex := 0,
                          ^.onFocus --> state.modState(State.isInterfaceFocused.set(true)),
                          ^.onBlur --> state.modState(State.isInterfaceFocused.set(false)),
                          ^.position := "relative",
                          MultiContigSpanHighlightableSentence(
                            MultiContigSpanHighlightableSentenceProps(
                              sentence = sentence.sentenceTokens,
                              styleForIndex = i =>
                              TagMod(Styles.specialWord, Styles.niceBlue).when(i == curVerbIndex),
                              highlightedSpans =
                                (inProgressAnswerOpt.map(_ -> (^.backgroundColor := "#FF8000")) ::
                                   curAnswers
                                   .map(_ -> (^.backgroundColor := "#FFFF00"))
                                   .map(Some(_))).flatten,
                              hover = hover(state.value.curFocus),
                              touch = touch(state.value.curFocus),
                              render = (
                                elements =>
                                <.p(Styles.largeText, Styles.unselectable, elements.toVdomArray)
                              )
                            )
                          ),
                          <.div(
                            liveTextField(
                              Some("Clause inclusion threshold:"),
                              clauseInclusionThreshold,
                              s => scala.util.Try(s.toDouble).toOption
                            )
                          ),
                          <.div(
                            liveTextField(
                              Some("Question inclusion threshold:"),
                              questionInclusionThreshold,
                              s => scala.util.Try(s.toDouble).toOption
                            )
                          ),
                          <.div(
                            liveTextField(
                              Some("Span inclusion threshold:"),
                              spanInclusionThreshold,
                              s => scala.util.Try(s.toDouble).toOption
                            )
                          ),
                          <.div(
                            checkboxToggle(
                              label = "Show discourse questions",
                              isValueActive = showDiscourseQuestions
                            )
                          ),
                          <.div(
                            ^.paddingTop := "20px",
                            clauseGroups.zipWithIndex
                              .filter(_._1.aggregateProb >= clauseInclusionThreshold.value)
                              .toVdomArray { case (clauseGroup, clauseIndex) =>
                                val slotsWithProbs = clauseGroup.slotProbs.toList.sortBy(-_._2)
                                val argValues = clauseGroup.bestSpansAboveThreshold(spanInclusionThreshold.value).map { case (slot, span) =>
                                  slot -> Text.renderSpan(sentence.sentenceTokens, span)
                                }
                                val mainClause = clauseGroup.frame.clausesWithArgs(argValues).head

                                val tenseLens = Frame.tan.composeLens(TAN.tense)
                                val negLens = Frame.tan.composeLens(TAN.isNegated)

                                val ngClause = tenseLens
                                  .set(Tense.NonFinite.Gerund)(clauseGroup.frame)
                                  .clausesWithArgs(argValues).head
                                val negFlippedNgClause = tenseLens
                                  .set(Tense.NonFinite.Gerund)(
                                    negLens.modify(!_)(clauseGroup.frame)
                                  )
                                  .clausesWithArgs(argValues).head
                                val toClause = tenseLens
                                  .set(Tense.NonFinite.To)(clauseGroup.frame)
                                  .clausesWithArgs(argValues).head
                                val invClause = clauseGroup.frame
                                  .questionsForSlotWithArgs(None, argValues).head.init
                                val invNegWouldClause = tenseLens
                                  .set(Tense.Finite.Modal("would".lowerCase))(
                                    negLens.set(true)(clauseGroup.frame)
                                  )
                                  .questionsForSlotWithArgs(None, argValues).head.init

                                <.div(
                                  <.h5(f"(${clauseGroup.aggregateProb}%.2f) $mainClause"),
                                  <.ul(
                                    ^.classSet1("list-unstyled"),
                                    slotsWithProbs.zipWithIndex
                                      .filter(_._1._2 >= questionInclusionThreshold.value)
                                      .toVdomArray { case ((slot, prob), questionIndex) =>
                                        val questionString = clauseGroup.frame.questionsForSlotWithArgs(slot, argValues).head
                                          <.li(
                                            ^.key := s"question-$clauseIndex-$questionIndex",
                                            ^.display := "block",
                                            qaField(state, sentence.sentenceTokens, clauseGroup.verbIndex, questionString, prob, highlightedAnswers)(
                                              (clauseIndex, questionIndex)
                                            ),
                                            <.div(
                                              ^.paddingLeft := "48px",
                                              clauseGroup.slotSpans(slot).filter(_._2 >= spanInclusionThreshold.value).sortBy(-_._2).map {
                                                case (span, prob) => Text.renderSpan(sentence.sentenceTokens, span) +
                                                    f" ($prob%.2f)"
                                              }.mkString(" / ")
                                            )
                                          )
                                      }
                                  ),
                                  <.div(
                                    <.h5("Discourse questions:"),
                                    <.ul(
                                      ^.classSet1("list-unstyled"),
                                      ^.paddingLeft := "24px",
                                      <.li(s"What is the result of $ngClause?"),
                                      <.li(s"What is the cause of $ngClause?"),
                                      <.li(s"What is the reason that $mainClause?"),
                                      <.li(s"Under what condition $invClause?"),
                                      <.li(s"Under what condition $invNegWouldClause?"),
                                      <.li(s"In what manner $invClause?"),
                                      <.li(s"What is the exception to $ngClause?"),
                                      <.li(s"What happens at the same time as $ngClause?"),
                                      <.li(s"What happens before $ngClause?"),
                                      <.li(s"What happens after $ngClause?"),
                                      <.li(s"What is an example of $ngClause?"),
                                      <.li(s"Despite what $invClause?"),
                                      <.li(s"What is contrasted with $ngClause?"),
                                      <.li(s"What is similar to $ngClause?"),
                                      <.li(s"What is an alternative to $ngClause?"),
                                      <.li(s"What is an alternative to $negFlippedNgClause?"),
                                      <.li(s"What are the options for $ngClause?")
                                    )
                                  ).when(showDiscourseQuestions.value)
                                )
                              }
                          )
                        )
                          // <.div(
                          //   ^.classSet1("form-group"),
                          //   ^.margin := "5px",
                          //   <.textarea(
                          //     ^.classSet1("form-control"),
                          //     ^.name := FieldLabels.feedbackLabel,
                          //     ^.rows := 3,
                          //     ^.placeholder := "Feedback? (Optional)"
                          //   )
                          // ),
                          // <.input(
                          //   ^.classSet1("btn btn-primary btn-lg btn-block"),
                          //   ^.margin := "5px",
                          //   ^.`type` := "submit",
                          //   ^.disabled := !answers.forall(_.forall(_.isComplete)),
                          //   ^.id := FieldLabels.submitButtonLabel,
                          //   ^.value := (
                          //     if (isNotAssigned) "You must accept the HIT to submit results"
                          //     else if (!answers.forall(_.forall(_.isComplete)))
                          //       "You must respond to all questions to submit results"
                          //     else "Submit"
                          //   )
                          // )
                      )
                  }
                }
              }
            }
          }
        }
    }
  }

  val FullUI = ScalaComponent
    .builder[Unit]("Full UI")
    .render(_ => body)
    // .componentDidUpdate(_.backend.updateResponse) // TODO make response update properly
    .build

}
