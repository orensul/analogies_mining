package qfirst.frames.annotation

import cats.Id
import cats.Order
import cats.data.NonEmptyList
import cats.implicits._

import scalajs.js
import org.scalajs.dom
import org.scalajs.dom.html
import org.scalajs.dom.ext.KeyCode

import scala.concurrent.ExecutionContext.Implicits.global

import japgolly.scalajs.react.vdom.html_<^._
import japgolly.scalajs.react._
import japgolly.scalajs.react.extra.StateSnapshot

import scalacss.DevDefaults._
import scalacss.ScalaCssReact._

import monocle._
import monocle.function.{all => Optics}
import monocle.macros._
import japgolly.scalajs.react.MonocleReact._

import qasrl.bank.AnswerSource
import qasrl.bank.AnnotationRound
import qasrl.bank.DataIndex
import qasrl.bank.DatasetPartition
import qasrl.bank.Document
import qasrl.bank.DocumentId
import qasrl.bank.DocumentMetadata
import qasrl.bank.Domain
import qasrl.bank.QuestionSource
import qasrl.bank.SentenceId

import qasrl.bank.service.DocumentService

import qasrl.data.AnswerLabel
import qasrl.data.AnswerJudgment
import qasrl.data.Answer
import qasrl.data.InvalidQuestion
import qasrl.data.Sentence
import qasrl.data.VerbEntry
import qasrl.data.QuestionLabel

import jjm.LowerCaseString
import jjm.OrWrapped
import jjm.ling.ESpan
import jjm.ling.Text
import jjm.ui._
import jjm.implicits._

import scala.collection.immutable.SortedSet

import io.circe._

import scala.concurrent.Future

case class Rgba(r: Double, g: Double, b: Double, a: Double) {
  def add(that: Rgba) = {
    if(this.a == 0.0) that else if(that.a == 0.0) this else {
      val alpha = 1.0 - ((1.0 - a) * (1.0 - that.a))
      Rgba(
        (a * r / alpha) + ((1.0 - a) * (that.r * that.a) / alpha),
        (a * g / alpha) + ((1.0 - a) * (that.g * that.a) / alpha),
        (a * b / alpha) + ((1.0 - a) * (that.b * that.a) / alpha),
        alpha
      )
    }
  }
  def toColorStyleString = f"rgba(${math.round(r)}%d, ${math.round(g)}%d, ${math.round(b)}%d, $a%.4f)"
}

object FrameAnnClient {

  val S = FrameAnnStyles

  case class Props(
    docService: DocumentService[OrWrapped[AsyncCallback, ?]],
    annService: ClauseAnnotationService[AsyncCallback]
  )

  case class State()
  object State {
    val initial = State()
  }

  @Lenses case class ResId(
    isFull: Boolean,
    index: Int) {
    def toProxy = ResIdProxy(isFull, index.toString)
  }

  @Lenses case class ResIdProxy(
    isFull: Boolean,
    index: String) {
    def toResId = scala.util.Try(index.toInt).toOption.map(ResId(isFull, _))
  }

  val ResIdLocal = new LocalState[ResId]
  val ResIdProxyLocal = new LocalState[ResIdProxy]
  val ResLocal = new LocalState[ClauseResolution]
  val ResFetch = new CacheCallContent[ResId, ClauseResolution]
  val DocFetch = new CacheCallContent[DocumentId, Document]

  def checkboxToggle[A](
    label: String,
    isValueActive: StateSnapshot[Boolean]
  ) = <.div(
    <.input(S.checkbox)(
      ^.`type` := "checkbox",
      ^.value := label,
      ^.checked := isValueActive.value,
      ^.onChange --> isValueActive.modState(!_)
    ),
    <.span(S.checkboxLabel)(
      label
    )
  )

  val transparent = Rgba(255, 255, 255, 0.0)
  val queryKeywordHighlightLayer = Rgba(255, 255, 0, 0.4)

  val highlightLayerColors = List(
    // Rgba(255, 255,   0, 0.2), // yellow
    Rgba(  0, 128, 255, 0.1), // green-blue
    Rgba(255,   0, 128, 0.1), // magenta?
    Rgba( 64, 192,   0, 0.1), // something. idk
    Rgba(128,   0, 255, 0.1), // mystery
    Rgba(  0, 255, 128, 0.1)  // blue-green
  )

  // val helpModalId = "help-modal"
  // val helpModalLabelId = "help-modal-label"
  // val dataToggle = VdomAttr("data-toggle")
  // val dataTarget = VdomAttr("data-target")
  // val ariaLabelledBy = VdomAttr("aria-labelledby")
  // val ariaHidden = VdomAttr("aria-hidden")
  // val dataDismiss = VdomAttr("data-dismiss")
  // val ariaLabel = VdomAttr("aria-label")

  import cats.Order.catsKernelOrderingForOrder

  sealed trait SpanColoringSpec {
    def spansWithColors: List[(ESpan, Rgba)]
  }
  case class RenderWholeSentence(val spansWithColors: List[(ESpan, Rgba)]) extends SpanColoringSpec
  case class RenderRelevantPortion(spansWithColorsNel: NonEmptyList[(ESpan, Rgba)]) extends SpanColoringSpec {
    def spansWithColors = spansWithColorsNel.toList
  }

  def renderSentenceWithHighlights(
    sentenceTokens: Vector[String],
    coloringSpec: SpanColoringSpec,
    wordRenderers : Map[Int, VdomTag => VdomTag] = Map()
  ) = {
    val containingSpan = coloringSpec match {
      case RenderWholeSentence(_) =>
        ESpan(0, sentenceTokens.size)
      case RenderRelevantPortion(swcNel) =>
        val spans = swcNel.map(_._1)
        ESpan(spans.map(_.begin).minimum, spans.map(_.end).maximum)
    }
    val wordIndexToLayeredColors = (containingSpan.begin until containingSpan.end).map { i =>
      i -> coloringSpec.spansWithColors.collect {
        case (span, color) if span.contains(i) => color
      }
    }.toMap
    val indexAfterToSpaceLayeredColors = ((containingSpan.begin + 1) to containingSpan.end).map { i =>
      i -> coloringSpec.spansWithColors.collect {
        case (span, color) if span.contains(i - 1) && span.contains(i) => color
      }
    }.toMap
    Text.renderTokens[Int, List, List[VdomElement]](
      words = sentenceTokens.indices.toList,
      getToken = (index: Int) => sentenceTokens(index),
      spaceFromNextWord = (nextIndex: Int) => {
        if(!containingSpan.contains(nextIndex) || nextIndex == containingSpan.begin) List() else {
          val colors = indexAfterToSpaceLayeredColors(nextIndex)
          val colorStr = NonEmptyList[Rgba](transparent, colors)
            .reduce((x: Rgba, y: Rgba) => x add y).toColorStyleString
          List(
            <.span(
              ^.key := s"space-$nextIndex",
              ^.style := js.Dynamic.literal("backgroundColor" -> colorStr),
              " "
            )
          )
        }
      },
      renderWord = (index: Int) => {
        if(!containingSpan.contains(index)) List() else {
          val colorStr = NonEmptyList(transparent, wordIndexToLayeredColors(index))
            .reduce((x: Rgba, y: Rgba) => x add y).toColorStyleString
          val render: (VdomTag => VdomTag) = wordRenderers.get(index).getOrElse((x: VdomTag) => x)
          val element: VdomTag = render(
            <.span(
              ^.style := js.Dynamic.literal("backgroundColor" -> colorStr),
              Text.normalizeToken(sentenceTokens(index))
            )
          )
          List(element(^.key := s"word-$index"))
        }
      }
    ).toVdomArray(x => x)
  }

  // def makeAllHighlightedAnswer(
  //   sentenceTokens: Vector[String],
  //   spans: NonEmptyList[ESpan],
  //   color: Rgba
  // ): VdomArray = {
  //   val orderedSpans = spans.sorted
  //   case class GroupingState(
  //     completeGroups: List[NonEmptyList[ESpan]],
  //     currentGroup: NonEmptyList[ESpan]
  //   )
  //   val groupingState = orderedSpans.tail.foldLeft(GroupingState(Nil, NonEmptyList.of(orderedSpans.head))) {
  //     case (GroupingState(groups, curGroup), span) =>
  //       if(curGroup.exists(s => s.overlaps(span))) {
  //         GroupingState(groups, span :: curGroup)
  //       } else {
  //         GroupingState(curGroup :: groups, NonEmptyList.of(span))
  //       }
  //   }
  //   val contigSpanLists = NonEmptyList(groupingState.currentGroup, groupingState.completeGroups)
  //   val answerHighlighties = contigSpanLists.reverse.map(spanList =>
  //     List(
  //       <.span(
  //         renderSentenceWithHighlights(sentenceTokens, RenderRelevantPortion(spanList.map(_ -> color)))
  //       )
  //     )
  //   ).intercalate(List(<.span(" / ")))
  //   answerHighlighties.zipWithIndex.toVdomArray { case (a, i) =>
  //     a(^.key := s"answerString-$i")
  //   }
  // }

  // val colspan = VdomAttr("colspan")


  class Backend(scope: BackendScope[Props, State]) {

    def render(props: Props, state: State) = {
      ResIdLocal.make(initialValue = ResId(false, 0)) { resId =>
        <.div(S.mainContainer)(
          ResIdProxyLocal.make(initialValue = resId.value.toProxy) { resIdProxy =>
            <.div(S.fixedRowContainer, S.headyContainer)(
              checkboxToggle("Full ambiguity", resIdProxy.zoomStateL(ResIdProxy.isFull)),
              <.input(
                ^.`type` := "text",
                ^.placeholder := "Index of ambiguity",
                ^.value := resIdProxy.value.index,
                ^.onChange ==> ((e: ReactEventFromInput) => resIdProxy.zoomStateL(ResIdProxy.index).setState(e.target.value)),
                ^.onKeyDown ==> (
                  (e: ReactKeyboardEvent) => {
                    CallbackOption.keyCodeSwitch(e) {
                      case KeyCode.Enter =>
                        resIdProxy.value.toResId.fold(Callback.empty)(resId.setState)
                    }
                  }
                )
              ),
              <.button(
                ^.`type` := "button",
                ^.onClick --> resId.zoomStateL(ResId.index).modState(_ - 1),
                "<--"
              ),
              <.button(
                ^.`type` := "button",
                ^.onClick --> resId.zoomStateL(ResId.index).modState(_ + 1),
                "-->"
              )
            )
          },
          ResFetch.make(request = resId.value, sendRequest = id => OrWrapped.wrapped(props.annService.getResolution(id.isFull, id.index))) {
            case ResFetch.Loading => <.div(S.loadingNotice)("Waiting for clause ambiguity data...")
            case ResFetch.Loaded(loadedClauseResolution) =>
              ResLocal.make(initialValue = loadedClauseResolution) { clauseResolutionS =>
                val clauseResolution = clauseResolutionS.value
                val ambig = clauseResolution.ambiguity
                val sid = ambig.sentenceId
                DocFetch.make(request = sid.documentId, sendRequest = id => props.docService.getDocument(id)) {
                  case DocFetch.Loading => <.div(S.loadingNotice)("Waiting for document...")
                  case DocFetch.Loaded(document) =>
                    val sentence = document.sentences.find(_.sentenceId == SentenceId.toString(sid)).get
                    val verbEntry = sentence.verbEntries(ambig.verbIndex)
                    val questionLabel = verbEntry.questionLabels(ambig.questionString)

                    val blueGreen = Rgba(0, 128, 255, 0.1)
                    val verbColorMap = Map(verbEntry.verbIndex -> blueGreen)
                    val answerSpansWithColors = questionLabel.answerJudgments.toList
                      .flatMap(_.judgment.getAnswer).flatMap(_.spans.toList)
                      .map(span => span -> blueGreen)

                    <.div(S.flexyBottomContainer, S.flexColumnContainer)(
                      <.div(S.fixedRowContainer, S.sentenceTextContainer, S.headyContainer)(
                        <.span(S.sentenceText)(
                          renderSentenceWithHighlights(
                            sentence.sentenceTokens,
                            RenderWholeSentence(answerSpansWithColors),
                            verbColorMap.collect { case (verbIndex, color) =>
                              verbIndex -> (
                                (v: VdomTag) => <.a(
                                  S.verbAnchorLink,
                                  ^.href := s"#verb-$verbIndex",
                                  v(
                                    ^.color := color.copy(a = 1.0).toColorStyleString,
                                    ^.fontWeight := "bold",
                                    )
                                )
                              )
                            }
                          )
                        )
                      ),
                      <.div(S.verbEntryDisplay)(
                        <.div(S.verbHeading)(
                          <.span(S.verbHeadingText)(
                            ^.color := blueGreen.copy(a = 1.0).toColorStyleString,
                            sentence.sentenceTokens(verbEntry.verbIndex)
                          )
                        ),
                        <.div(S.questionHeading)(
                          <.span(S.questionHeadingText)(
                            ambig.questionString
                          )
                        ),
                        <.table(S.verbQAsTable)(
                          <.tbody(S.verbQAsTableBody)(
                            ambig.structures.toList.toVdomArray { clauseChoice =>
                              val innerCell = <.td(
                                <.span(S.clauseChoiceText)(
                                  clauseChoice.frame.clauses(true).mkString(" / "),
                                  ),
                                ^.onClick --> (
                                  Callback(println(clauseChoice)) >> {
                                      val curChoice = clauseResolution.choiceOpt.getOrElse(Set.empty[ClauseChoice])
                                      val newChoice = if(curChoice.contains(clauseChoice)) curChoice - clauseChoice else curChoice + clauseChoice
                                      props.annService.saveResolution(resId.value.isFull, resId.value.index, newChoice)
                                        .flatMap(s => clauseResolutionS.setState(s).asAsyncCallback)
                                        .toCallback
                                    }
                                )
                              )
                              if(clauseResolution.choiceOpt.exists(_.contains(clauseChoice))) {
                                <.tr(S.clauseChoiceRow, S.darkerClauseChoiceRow)(
                                  innerCell
                                )
                              } else {
                                <.tr(S.clauseChoiceRow)(
                                  innerCell
                                )
                              }
                            },
                            {
                              val innerCell = <.td(
                                <.span(S.clauseChoiceText)(
                                  "<None>"
                                ),
                                ^.onClick --> (
                                  props.annService.saveResolution(resId.value.isFull, resId.value.index, Set.empty[ClauseChoice])
                                    .flatMap(s => clauseResolutionS.setState(s).asAsyncCallback)
                                    .toCallback
                                )
                              )
                              if(clauseResolution.choiceOpt.exists(_.isEmpty)) {
                                <.tr(S.clauseChoiceRow, S.darkerClauseChoiceRow)(innerCell)
                              } else {
                                <.tr(S.clauseChoiceRow)(innerCell)
                              }
                            }
                          )
                        ),
                        <.div(^.paddingTop := "20px")(
                          <.h3("Other Questions"),
                          <.ul(
                            verbEntry.questionLabels.keySet.toList.toVdomArray(qStr =>
                              <.li(qStr)
                            )
                          )
                        )
                      )
                    )
                }
              }
          }
        )
      }
    }
  }

  val Component = ScalaComponent.builder[Props]("FrameAnnClient")
    .initialState(State.initial)
    .renderBackend[Backend]
    .build

}
