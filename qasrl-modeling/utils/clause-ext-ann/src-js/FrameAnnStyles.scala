package qfirst.frames.annotation

import scalacss.DevDefaults._
import scala.language.postfixOps

object FrameAnnStyles extends StyleSheet.Inline {
  import dsl._

  // color scheme

  val headerBackgroundColor = grey(240)
  // val headerContentColor = black

  // val selectedHighlightColor = grey(200)
  // val hoverHighlightColor = grey(240)
  val alternatingRowBackgroundColor1 = white
  val alternatingRowBackgroundColor2 = grey(240)

  // val originalRoundIndicatorColor = grey(200)
  // val expansionRoundIndicatorColor = rgba(  64, 192,   0, 1.0)
  // val evalRoundIndicatorColor = orange

  // val metadataLabelBackgroundColor = grey(240)

  // val validTextColor = green
  // val invalidTextColor = red

  // val paneDivisionBorderWidth = 1 px
  // val paneDivisionBorderColor = metadataLabelBackgroundColor

  // val sentenceSelectionPaneBorder = style(
  //   borderLeftColor(paneDivisionBorderColor),
  //   borderRightColor(paneDivisionBorderColor),
  //   borderLeftStyle.solid,
  //   borderRightStyle.solid,
  //   borderLeftWidth(paneDivisionBorderWidth),
  //   borderRightWidth(paneDivisionBorderWidth)
  // )

  // styles

  val checkbox = style(
    addClassNames("form-check-input")
  )

  val checkboxLabel = style(
    addClassNames("form-check-label")
  )

  // val webkitScrollbar = {
  //   import scalacss.internal._
  //   Cond(Some(Pseudo.Custom("::-webkit-scrollbar", PseudoType.Element)), Vector.empty)
  // }

  // actual styles

  val flexColumnContainer = style(
    overflow.hidden,
    backfaceVisibility.hidden,
    willChange := "overflow",
    display.flex,
    flexDirection.column
  )

  val mainContainer = style(
    addClassNames("container-fluid"),
    flexColumnContainer,
    position.relative,
    height(100 vh)
  )

  // val mainTitle = style()

  val fixedRowContainer = style(
    addClassNames("px-3", "py-1"),
    display.block,
    width(100 %%)
  )

  val queryInputContainer = style(
    // fixedRowContainer,
    display.flex,
    flexDirection.row
  )

  val queryInput = style(
    flex := "1"
  )
  // val querySubmitButton = style()

  val flexyBottomContainer = style(
    flex := "1"
  )

  // val scrollPane = style(
  //   overflow.auto,
  //   height.auto,
  //   webkitScrollbar(
  //     display.none
  //   )
  //     // attr("-webkit-overflow-scrolling") := "touch",
  //     // attr("-ms-overflow-style") := "none"
  // )

  val headyContainer = style(
    backgroundColor(headerBackgroundColor)
  )



  // // main data display

  // val dataContainer = style(
  //   position.relative,
  //   overflow.hidden,
  //   backfaceVisibility.hidden,
  //   willChange := "overflow",
  //   display.flex,
  //   // marginTop(-headerHeight),
  //   // paddingTop(headerHeight),
  //   height(100 vh),
  //   width(100 %%)
  // )

  // // selection of sentences

  // val metadataLabelHeight = 1 rem
  // val metadataLabelFontSize = 8 pt

  // val metadataLabel = style(
  //   display.block,
  //   height(metadataLabelHeight),
  //   backgroundColor(metadataLabelBackgroundColor),
  //   fontSize(metadataLabelFontSize),
  //   verticalAlign.middle
  // )
  // val metadataLabelText = style(
  //   addClassNames("px-1"),
  //   whiteSpace.nowrap
  // )

  // val documentSelectionPaneWidth = 10 rem
  // val sentenceSelectionPaneWidth = 12 rem

  // val documentSelectionFontSize = 12 pt
  // val sentenceSelectionFontSize = 10 pt

  // val contentPaneContainer = style(
  //   position.relative,
  //   overflow.hidden,
  //   backfaceVisibility.hidden,
  //   willChange := "overflow",
  //   display.flex,
  //   flexDirection.column
  // )

  // val selectionPane = style(
  //   scrollPane,
  //   lineHeight(1.2)
  // )

  // val countLabel = style(
  //   metadataLabel,
  //   textAlign.right
  // )
  // val countLabelText = style(
  //   metadataLabelText
  // )

  // val selectionEntry = style(
  //   addClassNames("p-2"),
  //   &.hover(
  //     backgroundColor(hoverHighlightColor)
  //   )
  // )

  // val currentSelectionEntry = style(
  //   selectionEntry,
  //   backgroundColor(selectedHighlightColor).important
  // )

  // val nonCurrentSelectionEntry = style(
  //   selectionEntry
  // )

  // val documentSelectionPaneContainer = style(
  //   contentPaneContainer,
  //   width(documentSelectionPaneWidth),
  // )

  // val documentCountLabel = style(
  //   countLabel
  // )

  // val documentCountLabelText = style(
  //   countLabelText
  // )

  // val documentSelectionPane = style(
  //   selectionPane,
  //   width(100 %%)
  // )

  // val documentSelectionEntry = style(
  //   selectionEntry
  // )

  // val documentSelectionEntryText = style(
  //   fontSize(documentSelectionFontSize)
  // )

  // val sentenceSelectionPaneContainer = style(
  //   contentPaneContainer,
  //   sentenceSelectionPaneBorder,
  //   width(sentenceSelectionPaneWidth)
  // )

  // val sentenceCountLabel = style(
  //   countLabel
  // )

  // val sentenceCountLabelText = style(
  //   countLabelText
  // )

  // val sentenceSelectionPane = style(
  //   selectionPane,
  // )

  // val sentenceSelectionEntry = style(
  //   selectionEntry,
  // )

  // val sentenceSelectionEntryText = style(
  //   fontSize(sentenceSelectionFontSize),
  // )

  // // display of document biggy thing

  // val documentContainer = style(
  //   flex := "1",
  //   display.flex,
  //   flexDirection.row,
  //   overflow.hidden,
  //   position.relative,
  //   backfaceVisibility.hidden,
  //   willChange := "overflow"
  // )

  // // display of sentence data

  // val sentenceDisplayPane = style(
  //   contentPaneContainer,
  //   flex := "1"
  // )

  // val sentenceInfoContainer = style(
  //   addClassNames("pl-2"),
  //   metadataLabel,
  //   textAlign.left
  // )
  // val sentenceInfoText = style(
  //   metadataLabelText
  // )

  val sentenceTextContainer = style()

  val verbAnchorLink = style(
    &.hover(
      textDecoration := "none"
    )
  )

  // val verbEntriesContainer = style(
  //   scrollPane,
  //   flex := "1"
  // )

  val loadingNotice = style(
    addClassNames("p-3")
  )

  val sentenceText = style(
    fontSize(16 pt)
  )

  val verbEntryDisplay = style(
    addClassNames("px-4", "pb-4"),
    width(100 %%)
  )

  val verbHeading = style()

  val verbHeadingText = style(
    fontSize(16 pt),
    fontWeight.bold
  )

  val questionHeading = verbHeading
  val questionHeadingText = verbHeadingText

  val verbQAsTable = style(
    width(100 %%)
  )

  val verbQAsTableBody = style(
    width(100 %%)
  )

  // val hoverHighlightedVerbTable = style(
  //   backgroundColor(hoverHighlightColor)
  // )

  val clauseChoiceRow = style(
    addClassNames("p-1"),
    width(100 %%),
    // &.nthChild("odd")(
    //   backgroundColor(alternatingRowBackgroundColor1)
    // ),
    // &.nthChild("even")(
    //   backgroundColor(alternatingRowBackgroundColor2)
    // )
  )

  val clauseChoiceText = style(
    fontSize(16 pt),
    )

  val darkerClauseChoiceRow = style(
    backgroundColor(alternatingRowBackgroundColor2)
  )

  // val roundIndicator = style(
  //   width(0.2 rem),
  //   height(100 %%)
  // )

  // val originalRoundIndicator = style(
  //   roundIndicator,
  //   backgroundColor(originalRoundIndicatorColor)
  // )

  // val expansionRoundIndicator = style(
  //   roundIndicator,
  //   backgroundColor(expansionRoundIndicatorColor)
  // )

  // val evalRoundIndicator = style(
  //   roundIndicator,
  //   backgroundColor(evalRoundIndicatorColor)
  // )

  // // detour to legend real quick

  // val roundLegendMark = style(
  //   addClassNames("ml-2"),
  //   display.inlineBlock,
  //   color.transparent
  // )

  // val originalLegendMark = style(
  //   roundLegendMark,
  //   originalRoundIndicator
  // )

  // val expansionLegendMark = style(
  //   roundLegendMark,
  //   expansionRoundIndicator
  // )

  // val evalLegendMark = style(
  //   roundLegendMark,
  //   evalRoundIndicator
  // )

  // // back to table (question cells etc)

  // val questionCellPadding = style(
  //   addClassNames("pl-1"),
  // )

  // val questionCell = style(
  //   questionCellPadding,
  //   width(20 rem)
  // )
  // val questionText = style()

  // val validityCell = style(
  //   addClassNames("px-2"),
  //   width(2 rem)
  // )
  // val validityText = style()
  // val validValidityText = style(
  //   validityText,
  //   color(validTextColor)
  // )
  // val invalidValidityText = style(
  //   validityText,
  //   color(invalidTextColor)
  // )

  // val answerCell = style()
  // val answerText = style()

  // val qaPairDisplay = style()

  // val questionFullDescriptionCell = style(
  //   padding(0 px),
  //   margin(0 px)
  // )

  // val questionSourceText = style(
  //   questionCellPadding
  // )

  // val answerSourceIdCell = style(
  //   width(15 rem)
  // )

  // val dummyRow = style(
  //   margin(0 px),
  //   padding(0 px)
  // )
}
