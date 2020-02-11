import React, { useRef, useCallback, useContext, useEffect } from 'react';
import styled, { css } from 'styled-components';
import { Button } from '@allenai/varnish/components/button';

import { ChapterContext } from '../context';
import { Card, CardContent } from '../components/Card';
import { CheckMark } from '../components/inlineSVG/CheckMark';

const Exercise = ({ id, title, type, children }) => {
    const excRef = useRef();
    const excId = parseInt(id);
    const { activeExc, setActiveExc, completed, setCompleted } = useContext(ChapterContext);
    const isExpanded = activeExc === excId;
    const isCompleted = completed.includes(excId);
    useEffect(() => {
        if (isExpanded && excRef.current) {
            excRef.current.scrollIntoView();
        }
        document.addEventListener('keyup', handleEscape, false);
        return () => {
          document.removeEventListener('keyup', handleEscape, false);
        }
    }, [isExpanded]);
    const handleEscape = (e) => {
        if (e.keyCode === 27) {  // 27 is ESC
            setActiveExc(null);
        }
    };
    const handleExpand = useCallback(() => setActiveExc(isExpanded ? null : excId), [
        isExpanded,
        excId,
    ]);

    const handleNext = useCallback(() => setActiveExc(excId + 1));
    const handleSetCompleted = useCallback(() => {
        const newCompleted = isCompleted
            ? completed.filter(v => v !== excId)
            : [...completed, excId];
        setCompleted(newCompleted);
        if (!isCompleted) {
            setActiveExc(null);
        }
    }, [isCompleted, completed, excId]);

    const Title = isCompleted ? CompletedSectionTitle : SectionTitle;

    return (
        <StyledCard isExpanded={isExpanded}>
            <Anchor id={id}><div ref={excRef} /></Anchor>
            <Title onClick={handleExpand}>
                <SectionId>
                    {excId}
                </SectionId>
                {title}
                <CheckMark />
            </Title>
            {isExpanded && (
                <StyledCardContent>
                    <MarkdownContainer>
                        {children}
                        <Toolbar>
                            <Button onClick={handleSetCompleted}>
                                {isCompleted ? 'Unm' : 'M'}ark as Completed
                            </Button>
                            <NextButton onClick={handleNext}>
                                Next
                            </NextButton>
                        </Toolbar>
                    </MarkdownContainer>
                </StyledCardContent>
            )}
        </StyledCard>
    );
};

export default Exercise;

const NextButton = styled(Button)``;

const StyledCard = styled(({ isExpanded, ...props }) => <Card {...props} />)`
    border: 1px solid transparent;

    ${({ isExpanded, theme }) => !isExpanded ? `
        &:hover {
            border-color: ${theme.color.B6};
        }
    ` : null}

    // This is a hack to hide the last section "next" button.
    // I couldn't figure out how to get an array of Excercise components from
    // markdown. Excercise is not a child of chapter template, but part of
    // the markdown data that gets injected. I don't know of a way for an Excercise
    // component to self-identify as "last" without explicitly setting a "last" flag
    // in the markdown data itself. This solution hides the button from the user but does
    // not actually prevent the button from being rendered to the DOM which
    // would be prefarable. This is addressing an issue that was inherited from
    // https://github.com/ines/course-starter-python.
    // TODO: investigate possible ways of hiding this programmatically.

    &:last-child {
        ${NextButton} {
            display: none;
        }
    }
`;

// Workaround to position anchored content below sticky header
const Anchor = styled.div`
    width: 1px;
    height: 164px;
    margin-top: -164px;
    transform: translateX(-10px);
`;

const titleStyles = css`
    ${({ theme }) => theme.typography.bodyBig}
    display: flex;
    margin: 0;
    padding: ${({ theme }) => `${theme.spacing.lg} ${theme.spacing.md.getRemValue() * 2}rem`};
    cursor: pointer;

    svg {
        margin-left: auto;
        margin-right: -6px;
        opacity: 0;
    }
`;

const SectionTitle = styled.h2`
    ${titleStyles}
`;

const CompletedSectionTitle = styled.h2`
    ${titleStyles}
    
    svg {
        opacity: 1;
        fill: ${({ theme }) => theme.color.G6};
    }
`;

const SectionId = styled.span`
   font-weight: normal;
   color: ${({ theme }) => theme.color.B6};
   font-size: 19px;
   padding-right: 20px;
`;

const StyledCardContent = styled(CardContent)`
    border-top: 1px solid ${({ theme }) => theme.color.N4};
`;

const Toolbar = styled.div`
    border-top: 1px solid ${({ theme }) => theme.color.N4};
    padding-top: ${({ theme }) => theme.spacing.md.getRemValue() * 2}rem;
    margin-top: ${({ theme }) => theme.spacing.md.getRemValue() * 2}rem;
    display: flex;
    
    button:last-child {
      margin-left: auto;
    }
`;

const MarkdownContainer = styled.div`
    h1,
    h2,
    h3,
    h4 {
      ${({ theme }) => theme.typography.h4}
    }

    &&& {
        ol,
        ul {
            padding-left: 20px;
            margin: 1.5rem 0;
        }
    }

    ul {
      list-style: disc;
      
      ul {
        list-style: circle;
      }
    }
    
    a {
      text-decoration: none;
      
      &&:hover {
        text-decoration: underline;
      }
    }
    
    table,
    hr,
    pre,
    div[class^="code-module-root"],
    .gatsby-highlight {
        & + ${Toolbar}{
            border: none;
            padding-top: 0;
        }
    }
`;
